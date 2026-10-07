"""Locate the example event and write the numbers and figures used in the paper.

Two solutions are computed from the same picks:

  A. the workflow described in the paper, that is the released ambient-noise velocity model with the
     eikonal travel-time solver and Levenberg-Marquardt inversion of (x, y, z, t0);
  B. a homogeneous P velocity inverted jointly with the source, (x, y, z, Vp, t0).

The two solutions are complementary. Four of the seven stations stand on loose waste material above
the 338 m upper surface of the released model; solution B represents this near-surface layer and
solution A the deeper structure resolved by the survey. Their separation shows how the near-surface
velocity influences location estimates at this site.

The search volume is restricted to the area covered by the velocity survey, so neither solution
depends on the part of the grid where velocity is extrapolated beyond the survey boundary.
A jackknife over stations is reported as the practical location uncertainty.

Outputs, written to localization_outputs/:
    localization_summary.json   every number reported in the paper
    example_catalogue.csv       catalogue-style parameters of the example event
    fig5-c.pdf                  travel-time residuals per station, both solutions
    fig5-d.pdf                  stations, both solutions and their jackknife clouds

Usage: python run_localization_example.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import loc_xjd2 as loc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "XJD2_updated.txt"
OUT = HERE / "localization_outputs"
GRID_STEP_KM = (0.02, 0.02, 0.02)
PAD_KM = 0.3
Z_MIN_KM = 0.05


def survey_extent():
    arr = np.loadtxt(MODEL)
    return (((arr[:, 0].min() - loc.X_OFFSET_M) / 1e3, (arr[:, 0].max() - loc.X_OFFSET_M) / 1e3),
            (arr[:, 1].min() / 1e3, arr[:, 1].max() / 1e3),
            (Z_MIN_KM, arr[:, 2].max() / 1e3))


def subset(fields, keep):
    sub = loc.StationTravelTimeFields.__new__(loc.StationTravelTimeFields)
    sub.__dict__.update(fields.__dict__)
    sub.fields = [fields.fields[i] for i in keep]
    sub.tgrid = fields.tgrid[keep]
    sub.stations = fields.stations[keep]
    return sub


def locate_homogeneous(stas, d_obs, extent, vp_range=(0.2, 3.0)):
    """Grid search over (x, y, z, Vp) with the origin time solved analytically, then refined.

    The grid is halved three times around the current best point, which makes the result
    deterministic and independent of a starting guess.
    """
    (x0, x1), (y0, y1), (z0, z1) = extent
    step, vstep = 0.02, 0.02
    lo = np.array([x0, y0, z0, vp_range[0]])
    hi = np.array([x1, y1, z1, vp_range[1]])
    best = None
    for _ in range(4):
        gx = np.arange(lo[0], hi[0] + step, step)
        gy = np.arange(lo[1], hi[1] + step, step)
        gz = np.arange(lo[2], hi[2] + step, step)
        gv = np.arange(lo[3], hi[3] + vstep, vstep)
        X, Y = np.meshgrid(gx, gy, indexing="ij")
        best = (np.inf, None)
        for z in gz:
            d = np.sqrt((X[..., None] - stas[:, 0]) ** 2 + (Y[..., None] - stas[:, 1]) ** 2
                        + (z - stas[:, 2]) ** 2)
            for v in gv:
                tt = d / v
                t0 = (d_obs - tt).mean(axis=-1, keepdims=True)
                sse = ((d_obs - tt - t0) ** 2).sum(axis=-1)
                k = np.unravel_index(np.argmin(sse), sse.shape)
                if sse[k] < best[0]:
                    best = (float(sse[k]), (gx[k[0]], gy[k[1]], z, v, float(t0[k][0])))
        x, y, z, v, _ = best[1]
        lo = np.maximum([x - step, y - step, z - step, v - vstep], [x0, y0, z0, vp_range[0]])
        hi = np.minimum([x + step, y + step, z + step, v + vstep], [x1, y1, z1, vp_range[1]])
        step, vstep = step / 4, vstep / 4
    x, y, z, v, t0 = best[1]
    res = d_obs - (np.sqrt(((stas - [x, y, z]) ** 2).sum(axis=1)) / v + t0)
    return np.array([x, y, z, v, t0]), res


def main():
    OUT.mkdir(exist_ok=True)
    report = pd.read_csv(HERE / "code_inputs1" / "pick_report.csv", dtype={"sid": str})
    stas = np.load(HERE / "code_inputs1" / "stas_xyz.npy").astype(float)
    d_obs = np.load(HERE / "code_inputs1" / "tobs_p.npy").ravel()
    sids = report.sid.tolist()
    survey = survey_extent()
    arr = np.loadtxt(MODEL)
    z_top = arr[:, 2].max() / 1e3
    n_above = int((stas[:, 2] > z_top).sum())
    print(f"{len(stas)} stations, pick spread {d_obs.max() - d_obs.min():.3f} s")
    print(f"{n_above} of {len(stas)} stations stand above the top of the velocity survey ({z_top:.3f} km)")

    # A plane wave is the limiting case of a source far outside the array; comparing it with the
    # point-source fit shows whether the wavefront curvature across the array is measurable.
    c = stas[:, :2].mean(axis=0)
    A = np.column_stack([stas[:, :2] - c, np.ones(len(d_obs))])
    coef, *_ = np.linalg.lstsq(A, d_obs, rcond=None)
    rms_plane = float(np.sqrt(((d_obs - A @ coef) ** 2).mean()))

    print("\n=== Solution A: released velocity model, eikonal + Levenberg-Marquardt ===")
    vel_model = loc.load_velocity_model(str(MODEL), grid_step_km=GRID_STEP_KM, pad_km=PAD_KM)
    fields = loc.StationTravelTimeFields(vel_model, stas)
    hyc_a, cov_a, res_a = loc.locate(fields, d_obs, search_extent_km=survey)
    loc.present_loc_results(hyc_a, cov_a, res_a)
    jack_a = []
    for i in range(len(sids)):
        keep = [j for j in range(len(sids)) if j != i]
        t = d_obs[keep] - d_obs[keep].min()
        h, _, r = loc.locate(subset(fields, keep), t, search_extent_km=survey, verbose=False)
        jack_a.append((sids[i], h[:3], float(1000 * np.linalg.norm(h[:3] - hyc_a[:3]))))
        print(f"  without {sids[i]}: ({h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}) km, shift {jack_a[-1][2]:5.0f} m")

    print("\n=== Solution B: homogeneous velocity inverted jointly with the source ===")
    hyc_b, res_b = locate_homogeneous(stas, d_obs, survey)
    print(f"  loc ({hyc_b[0]:.4f}, {hyc_b[1]:.4f}, {hyc_b[2]:.4f}) km, Vp = {hyc_b[3]:.3f} km/s, "
          f"t0 = {hyc_b[4]:+.4f} s, RMS = {np.sqrt((res_b ** 2).mean()):.4f} s")
    jack_b = []
    for i in range(len(sids)):
        keep = [j for j in range(len(sids)) if j != i]
        t = d_obs[keep] - d_obs[keep].min()
        h, r = locate_homogeneous(stas[keep], t, survey)
        jack_b.append((sids[i], h[:3], float(1000 * np.linalg.norm(h[:3] - hyc_b[:3])), float(h[3])))
        print(f"  without {sids[i]}: ({h[0]:.3f}, {h[1]:.3f}, {h[2]:.3f}) km, Vp {h[3]:.2f} km/s, "
              f"shift {jack_b[-1][2]:5.0f} m")

    rms_a, rms_b = float(np.sqrt((res_a ** 2).mean())), float(np.sqrt((res_b ** 2).mean()))
    sd = np.sqrt(np.diag(cov_a))
    summary = {
        "event_class_label": "small landslide",
        "event_first_pick_utc": str(report.t_abs_p.min()),
        "station_ids": sids,
        "n_stations": len(sids),
        "pick_spread_s": round(float(d_obs.max() - d_obs.min()), 3),
        "velocity_survey_top_km": round(float(z_top), 3),
        "n_stations_above_survey_top": n_above,
        "search_extent_km": {"x": list(survey[0]), "y": list(survey[1]), "z": list(survey[2])},
        "plane_wave_rms_s": round(rms_plane, 4),
        "solution_A_released_velocity_model": {
            "location_km": {"x": round(float(hyc_a[0]), 4), "y": round(float(hyc_a[1]), 4),
                            "z": round(float(hyc_a[2]), 4)},
            "origin_time_s": round(float(hyc_a[3]), 4),
            "formal_1sigma": {"x_km": round(float(sd[0]), 4), "y_km": round(float(sd[1]), 4),
                              "z_km": round(float(sd[2]), 4), "t0_s": round(float(sd[3]), 4)},
            "residuals_s": [round(float(r), 4) for r in res_a],
            "residual_rms_s": round(rms_a, 4),
            "residual_max_abs_s": round(float(np.abs(res_a).max()), 4),
            "jackknife_max_shift_m": round(max(j[2] for j in jack_a)),
        },
        "solution_B_homogeneous_velocity": {
            "location_km": {"x": round(float(hyc_b[0]), 4), "y": round(float(hyc_b[1]), 4),
                            "z": round(float(hyc_b[2]), 4)},
            "vp_kms": round(float(hyc_b[3]), 3),
            "origin_time_s": round(float(hyc_b[4]), 4),
            "residuals_s": [round(float(r), 4) for r in res_b],
            "residual_rms_s": round(rms_b, 4),
            "residual_max_abs_s": round(float(np.abs(res_b).max()), 4),
            "jackknife_max_shift_m": round(max(j[2] for j in jack_b)),
            "jackknife_vp_range_kms": [round(min(j[3] for j in jack_b), 2),
                                       round(max(j[3] for j in jack_b), 2)],
        },
        "point_source_improvement_over_plane_wave": round(rms_plane / rms_b, 1),
        "separation_between_solutions_m": round(1000 * float(np.linalg.norm(hyc_a[:3] - hyc_b[:3]))),
    }
    (OUT / "localization_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def azimuthal_gap(src):
        az = np.sort(np.degrees(np.arctan2(stas[:, 0] - src[0], stas[:, 1] - src[1])) % 360)
        return float(np.max(np.diff(np.append(az, az[0] + 360))))

    def min_epicentral_m(src):
        return float(1000 * np.min(np.hypot(stas[:, 0] - src[0], stas[:, 1] - src[1])))

    jb = np.array([j[1] for j in jack_b])
    catalogue = pd.DataFrame([
        dict(solution="released_velocity_model", x_km=round(float(hyc_a[0]), 3), y_km=round(float(hyc_a[1]), 3),
             z_km=round(float(hyc_a[2]), 3), origin_time_utc=str((pd.Timestamp(report.t_abs_p.min()) + pd.Timedelta(seconds=float(hyc_a[3]))).round("ms")),
             n_stations=len(sids), n_picks=len(d_obs), azimuthal_gap_deg=round(azimuthal_gap(hyc_a), 1),
             min_epicentral_distance_m=round(min_epicentral_m(hyc_a)),
             horizontal_error_m=round(1000 * float(np.hypot(sd[0], sd[1]))), vertical_error_m=round(1000 * float(sd[2])),
             error_type="formal 1-sigma", rms_s=round(rms_a, 4)),
        dict(solution="homogeneous_velocity", x_km=round(float(hyc_b[0]), 3), y_km=round(float(hyc_b[1]), 3),
             z_km=round(float(hyc_b[2]), 3), origin_time_utc=str((pd.Timestamp(report.t_abs_p.min()) + pd.Timedelta(seconds=float(hyc_b[4]))).round("ms")),
             n_stations=len(sids), n_picks=len(d_obs), azimuthal_gap_deg=round(azimuthal_gap(hyc_b), 1),
             min_epicentral_distance_m=round(min_epicentral_m(hyc_b)),
             horizontal_error_m=round(1000 * float(np.max(np.hypot(jb[:, 0] - hyc_b[0], jb[:, 1] - hyc_b[1])))),
             vertical_error_m=round(1000 * float(np.max(np.abs(jb[:, 2] - hyc_b[2])))),
             error_type="station jackknife, largest shift", rms_s=round(rms_b, 4)),
    ])
    catalogue.to_csv(OUT / "example_catalogue.csv", index=False)
    print(catalogue.to_string(index=False))
    print(f"\nplane wave {1000 * rms_plane:.0f} ms | released model {1000 * rms_a:.0f} ms | "
          f"homogeneous {1000 * rms_b:.0f} ms")
    print(f"the two solutions are {summary['separation_between_solutions_m']} m apart")

    plt.rcParams.update({"font.family": "Times New Roman", "font.size": 11, "mathtext.fontset": "stix"})
    i = np.arange(len(sids))

    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    ax.plot(i, res_a, "-o", color="#1F4E9E", lw=2, ms=5, label=f"Released velocity model ({1000 * rms_a:.0f} ms)")
    ax.plot(i, res_b, "-s", color="#C0392B", lw=2, ms=5, label=f"Homogeneous velocity ({1000 * rms_b:.0f} ms)")
    ax.axhline(0, color="0.6", lw=0.8, ls=":")
    ax.set_xticks(i)
    ax.set_xticklabels([s[-3:] for s in sids])
    ax.set_xlabel("Station (last three digits of ID)")
    ax.set_ylabel("Travel-time residual (s)")
    ax.legend(frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-c.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5-c_preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.2, 3.7))
    ax.scatter(stas[:, 0], stas[:, 1], marker="^", s=120, c="#F0A050", ec="k", lw=0.8,
               label=f"Stations (n={len(stas)})", zorder=3)
    for (x, y, _), s in zip(stas, sids):
        ax.annotate(s[-3:], (x, y), xytext=(0, -14), textcoords="offset points", ha="center", fontsize=8)
    ax.scatter(hyc_a[0], hyc_a[1], marker="*", s=300, c="#1F4E9E", ec="k", lw=0.8,
               label=f"Released velocity model ({1000 * rms_a:.0f} ms)", zorder=5)
    ax.scatter(hyc_b[0], hyc_b[1], marker="*", s=300, c="#C0392B", ec="k", lw=0.8,
               label=f"Homogeneous velocity ({1000 * rms_b:.0f} ms)", zorder=5)
    ax.set_xlim(392.23, 392.88)
    ax.set_ylim(4865.73, 4866.04)
    ax.set_aspect("equal")
    ax.set_xlabel("Easting (km)")
    ax.set_ylabel("Northing (km)")
    ax.ticklabel_format(useOffset=False)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False, fontsize=9)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "fig5-d.pdf", bbox_inches="tight")
    fig.savefig(OUT / "fig5-d_preview.png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"figures and summary written to {OUT}")


if __name__ == "__main__":
    main()
