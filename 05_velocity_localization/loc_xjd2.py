"""
P-wave source localization (x, y, z, t0) in a three-dimensional velocity model.

Station-centred FMM fields provide reciprocal travel times and station-specific
source derivatives. A grid search supplies the initial position and origin time,
followed by Levenberg-Marquardt refinement with adaptive damping. Callers supply
the source search bounds. Coordinates and times are expressed in kilometres and
seconds; Z follows the elevation convention of the supplied input files.
"""
from typing import Optional, Sequence, Tuple

import numpy as np
import skfmm
from scipy.interpolate import RegularGridInterpolator, griddata

X_OFFSET_M = 32000000.0
VP_VS_RATIO = np.sqrt(3.0)  # Poisson ratio of 0.25.


# ======================================================================
# 1. Velocity model
# ======================================================================
def load_velocity_model(
        path_txt: str,
        grid_step_km: Tuple[float, float, float] = (0.02, 0.02, 0.02),
        x_offset_m: float = X_OFFSET_M,
        pad_km: float = 3.0,
        extent_km: Optional[Sequence[Tuple[float, float]]] = None,
        verbose: bool = True,
) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Read X(m), Y(m), Z(m), Vs(km/s) and interpolate onto a padded grid.

    pad_km: Padding distance on each side of the model.
    extent_km: Optional ((x0, x1), (y0, y1), (z0, z1)) bounds overriding padding.
    Velocities outside the interpolation domain are extended by nearest neighbour;
    report this assumption when stations or propagation paths use that region.
    """
    arr = np.loadtxt(path_txt)
    assert arr.shape[1] >= 4, "Input requires at least four columns: X, Y, Z, Vs."

    X_km = (arr[:, 0] - x_offset_m) / 1000.0
    Y_km = arr[:, 1] / 1000.0
    Z_km = arr[:, 2] / 1000.0
    Vp_kms = arr[:, 3] * VP_VS_RATIO

    if verbose:
        print(f"Loaded {len(X_km)} velocity samples.")
        print(f"Survey bounds: X [{X_km.min():.3f}, {X_km.max():.3f}] km, "
              f"Y [{Y_km.min():.3f}, {Y_km.max():.3f}] km, Z [{Z_km.min():.3f}, {Z_km.max():.3f}] km")
        print(f"Vp range: [{Vp_kms.min():.3f}, {Vp_kms.max():.3f}] km/s")

    dx_km, dy_km, dz_km = grid_step_km
    if extent_km is None:
        extent_km = ((X_km.min() - pad_km, X_km.max() + pad_km),
                     (Y_km.min() - pad_km, Y_km.max() + pad_km),
                     (Z_km.min() - pad_km, Z_km.max() + pad_km))
    (x0, x1), (y0, y1), (z0, z1) = extent_km

    x_coords_km = np.arange(x0, x1 + dx_km, dx_km)
    y_coords_km = np.arange(y0, y1 + dy_km, dy_km)
    z_coords_km = np.arange(z0, z1 + dz_km, dz_km)

    # Use linear interpolation where available and nearest neighbour elsewhere.
    if verbose:
        print("Interpolating the three-dimensional velocity grid...")
    points_km = np.vstack((X_km, Y_km, Z_km)).T
    Xg, Yg, Zg = np.meshgrid(x_coords_km, y_coords_km, z_coords_km, indexing="ij")
    query = np.array([Xg.ravel(), Yg.ravel(), Zg.ravel()]).T
    vg_lin = griddata(points_km, Vp_kms, query, method="linear")
    vg_nn = griddata(points_km, Vp_kms, query, method="nearest")
    vg = np.where(np.isfinite(vg_lin), vg_lin, vg_nn)

    # skfmm uses the axis order (z, y, x).
    vel_grid = vg.reshape(len(x_coords_km), len(y_coords_km), len(z_coords_km)).transpose(2, 1, 0)

    if verbose:
        n_out = int(np.sum(~np.isfinite(vg_lin)))
        print(f"Grid shape (Nz, Ny, Nx): {vel_grid.shape}; spacing (dx, dy, dz) = {grid_step_km} km")
        print(f"Nearest-neighbour extension supplies {n_out} grid points ({100 * n_out / len(vg):.1f}%) outside the interpolation domain.")

    return vel_grid, (z_coords_km, y_coords_km, x_coords_km)


def model_extent(vel_model) -> Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]:
    _, (z, y, x) = vel_model
    return (float(x.min()), float(x.max())), (float(y.min()), float(y.max())), (float(z.min()), float(z.max()))


def check_inside(vel_model, points: np.ndarray, what: str = "station") -> None:
    (x0, x1), (y0, y1), (z0, z1) = model_extent(vel_model)
    pts = np.atleast_2d(np.asarray(points, float))
    bad = ((pts[:, 0] < x0) | (pts[:, 0] > x1) | (pts[:, 1] < y0) | (pts[:, 1] > y1)
           | (pts[:, 2] < z0) | (pts[:, 2] > z1))
    if bad.any():
        raise ValueError(f"{int(bad.sum())} {what} point(s) outside the velocity grid: {pts[bad].tolist()}; "
                         f"grid bounds X [{x0:.3f}, {x1:.3f}], Y [{y0:.3f}, {y1:.3f}], Z [{z0:.3f}, {z1:.3f}] km. "
                         f"Expand pad_km or extent_km in load_velocity_model to include these points.")


# ======================================================================
# 2. Station travel-time fields by reciprocity
# ======================================================================
class StationTravelTimeFields:
    """Precompute one reciprocal travel-time field per station.

    Each station field provides its own source-position derivatives and can be
    reused across iterations and events with the same velocity grid and stations.
    """

    def __init__(self, vel_model, station_locs: np.ndarray, verbose: bool = True):
        check_inside(vel_model, station_locs, "station")
        vel, (z, y, x) = vel_model
        self.vel_model = vel_model
        self.z, self.y, self.x = z, y, x
        self.stations = np.asarray(station_locs, float)
        self.dz, self.dy, self.dx = (float(np.diff(a).mean()) for a in (z, y, x))

        Z, Y, X = np.meshgrid(z, y, x, indexing="ij")
        v_at = RegularGridInterpolator((z, y, x), vel)
        h = max(self.dx, self.dy, self.dz)

        self.fields = []
        for k, (sx, sy, sz) in enumerate(self.stations):
            dist = np.sqrt((X - sx) ** 2 + (Y - sy) ** 2 + (Z - sz) ** 2)
            # The analytic near-station radius must enclose a grid point so dist-r0 crosses zero.
            r0 = max(0.75 * h, 1.01 * float(dist.min()) + 1e-9)
            v0 = float(v_at([[sz, sy, sx]])[0])
            t = skfmm.travel_time(dist - r0, speed=vel, dx=(self.dz, self.dy, self.dx)) + r0 / v0
            t = np.where(dist <= r0, dist / v0, t)
            self.fields.append(RegularGridInterpolator((z, y, x), np.asarray(t, float)))
            if verbose:
                print(f"\rTravel-time field {k + 1}/{len(self.stations)}", end="")
        if verbose:
            print()
        self.tgrid = np.stack([np.asarray(f.values) for f in self.fields])

    def travel_times(self, source_loc) -> np.ndarray:
        """Return source-to-station travel times with shape (n_stations,)."""
        check_inside(self.vel_model, np.asarray(source_loc, float)[:3], "source")
        q = [[source_loc[2], source_loc[1], source_loc[0]]]
        return np.array([f(q)[0] for f in self.fields])

    def derivatives(self, source_loc) -> np.ndarray:
        """Return (dT/dx, dT/dy, dT/dz) with shape (n_stations, 3).

        Use half-grid central differences rather than storing full gradient grids.
        """
        p = np.asarray(source_loc, float)[:3]
        (x0, x1), (y0, y1), (z0, z1) = model_extent(self.vel_model)
        lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])
        h = 0.5 * np.array([self.dx, self.dy, self.dz])
        g = np.zeros((len(self.fields), 3))
        for j in range(3):
            a, b = p.copy(), p.copy()
            a[j] = min(p[j] + h[j], hi[j])
            b[j] = max(p[j] - h[j], lo[j])
            qa, qb = [[a[2], a[1], a[0]]], [[b[2], b[1], b[0]]]
            g[:, j] = [(f(qa)[0] - f(qb)[0]) / (a[j] - b[j]) for f in self.fields]
        return g


# ======================================================================
# 3. Initial position from grid search
# ======================================================================
def grid_search_init(fields: StationTravelTimeFields, d_obs: np.ndarray,
                     search_extent_km: Optional[Sequence[Tuple[float, float]]] = None,
                     verbose: bool = True) -> np.ndarray:
    """Search grid positions for the minimum squared arrival-time residual.

    Solve the origin time analytically as mean(d_obs - T) at each grid point.
    """
    if search_extent_km is None:
        search_extent_km = model_extent(fields.vel_model)
    (x0, x1), (y0, y1), (z0, z1) = search_extent_km
    iz = (fields.z >= z0) & (fields.z <= z1)
    iy = (fields.y >= y0) & (fields.y <= y1)
    ix = (fields.x >= x0) & (fields.x <= x1)
    if not (iz.any() and iy.any() and ix.any()):
        raise ValueError("The search bounds do not overlap the velocity grid.")

    tg = fields.tgrid[:, iz][:, :, iy][:, :, :, ix]
    d = d_obs.reshape(-1, 1, 1, 1)
    t0 = (d - tg).mean(axis=0)
    sse = ((d - tg - t0) ** 2).sum(axis=0)
    k = np.unravel_index(np.argmin(sse), sse.shape)
    hyc = np.array([fields.x[ix][k[2]], fields.y[iy][k[1]], fields.z[iz][k[0]], t0[k]])
    if verbose:
        print(f"[grid_search_init] Initial position: x={hyc[0]:.3f}, y={hyc[1]:.3f}, z={hyc[2]:.3f} km, t0={hyc[3]:.3f} s, "
              f"RMS={np.sqrt(sse[k] / len(d_obs)):.4f} s")
    return hyc


# ======================================================================
# 4. L-M localization iterations
# ======================================================================
def iter_loc_variable_velocity(hyc_loop, fields: StationTravelTimeFields, d_obs: np.ndarray,
                               niter: int = 50, lam: float = 1e-3,
                               search_extent_km: Optional[Sequence[Tuple[float, float]]] = None,
                               verbose: bool = True):
    """Refine the four source parameters (x, y, z, t0) using L-M.

    Return (hyc, cov, res): source parameters, covariance, and station residuals.
    """
    if search_extent_km is None:
        search_extent_km = model_extent(fields.vel_model)
    (x0, x1), (y0, y1), (z0, z1) = search_extent_km
    lo, hi = np.array([x0, y0, z0]), np.array([x1, y1, z1])

    hyc = np.asarray(hyc_loop, float).copy()
    hyc[:3] = np.clip(hyc[:3], lo, hi)
    sse = float(((d_obs - fields.travel_times(hyc) - hyc[3]) ** 2).sum())

    for k in range(niter):
        res = d_obs - fields.travel_times(hyc) - hyc[3]
        G = np.hstack([fields.derivatives(hyc), np.ones((len(d_obs), 1))])
        GTG = G.T @ G
        try:
            # Scale Marquardt damping by each parameter's diagonal curvature.
            step = np.linalg.solve(GTG + lam * np.diag(np.diag(GTG)), G.T @ res)
        except np.linalg.LinAlgError:
            if verbose:
                print(f"Iter {k + 1:2d}  Singular matrix; stopping iterations.")
            break

        trial = hyc + step
        trial[:3] = np.clip(trial[:3], lo, hi)
        sse_trial = float(((d_obs - fields.travel_times(trial) - trial[3]) ** 2).sum())
        if sse_trial < sse:
            moved = float(np.linalg.norm(trial[:3] - hyc[:3]))
            hyc, sse, lam = trial, sse_trial, max(lam * 0.3, 1e-9)
            if verbose:
                print(f"Iter {k + 1:2d}  SSE = {sse:.6e}  loc = ({hyc[0]:.4f}, {hyc[1]:.4f}, {hyc[2]:.4f})")
            if moved < 1e-5:
                if verbose:
                    print("Convergence criterion satisfied.")
                break
        else:
            lam *= 10.0
            if lam > 1e8:
                if verbose:
                    print("Damping became too large; stopping iterations.")
                break

    at_bound = (np.abs(hyc[:3] - lo) < 1e-9) | (np.abs(hyc[:3] - hi) < 1e-9)
    if at_bound.any() and verbose:
        print(f"Warning: the solution is on a search boundary (XYZ indices {np.where(at_bound)[0].tolist()}); "
              f"these coordinates are not resolved by the data and should not be reported as a resolved source location.")

    res = d_obs - fields.travel_times(hyc) - hyc[3]
    G = np.hstack([fields.derivatives(hyc), np.ones((len(d_obs), 1))])
    dof = max(len(d_obs) - 4, 1)
    # Use the undamped normal matrix for covariance to avoid understating uncertainty.
    cov = float(res @ res / dof) * np.linalg.pinv(G.T @ G)
    return hyc, cov, res


def locate(fields: StationTravelTimeFields, d_obs: np.ndarray,
           search_extent_km: Optional[Sequence[Tuple[float, float]]] = None,
           niter: int = 50, lam: float = 1e-3, verbose: bool = True):
    """Initialize by grid search and refine using L-M iterations."""
    hyc_init = grid_search_init(fields, d_obs, search_extent_km, verbose=verbose)
    return iter_loc_variable_velocity(hyc_init, fields, d_obs, niter=niter, lam=lam,
                                      search_extent_km=search_extent_km, verbose=verbose)


# ======================================================================
# 5. Result reporting
# ======================================================================
def present_loc_results(hyc, cov=None, res=None):
    x, y, z, t0 = hyc
    if cov is None:
        print(f"\nSource location: x={x:.3f}, y={y:.3f}, z={z:.3f} km, t0={t0:.3f} s")
    else:
        std = np.sqrt(np.diag(cov))
        print("\nSource location (+/-1 sigma):")
        print(f"  x  = {x:.4f} ± {std[0]:.4f} km")
        print(f"  y  = {y:.4f} ± {std[1]:.4f} km")
        print(f"  z  = {z:.4f} ± {std[2]:.4f} km")
        print(f"  t0 = {t0:.4f} ± {std[3]:.4f} s")
    if res is not None:
        print(f"  Travel-time residual: RMS = {np.sqrt((res ** 2).mean()):.4f} s, "
              f"mean = {res.mean():.4f} s, maximum absolute = {np.abs(res).max():.4f} s")
