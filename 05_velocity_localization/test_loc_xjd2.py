"""Verification of the localization code in loc_xjd2.py.

Test 1 checks the travel-time derivatives: the analytic partial derivatives with respect to source
        position are compared with finite differences of the computed travel times, and the rank of
        the station Jacobian is reported. A rank below three means the source position cannot be
        resolved, whatever the data.
Test 2 recovers synthetic sources whose positions are known by construction: travel times are
        computed with the same eikonal solver, perturbed by pick noise, and inverted.
Test 3 reports the solution for the released picks in code_inputs1/.

Usage: python test_loc_xjd2.py
"""
from pathlib import Path

import numpy as np

import loc_xjd2 as loc

HERE = Path(__file__).resolve().parent
MODEL = HERE / "XJD2_updated.txt"
PICK_NOISE_S = 0.01
SEED = 0
SYNTHETIC_SOURCES = ([392.70, 4865.75, 0.25], [392.55, 4865.90, 0.15], [392.85, 4865.65, 0.30],
                     [392.45, 4865.70, 0.20], [392.95, 4865.85, 0.10])


def survey_extent():
    arr = np.loadtxt(MODEL)
    return (((arr[:, 0].min() - loc.X_OFFSET_M) / 1e3, (arr[:, 0].max() - loc.X_OFFSET_M) / 1e3),
            (arr[:, 1].min() / 1e3, arr[:, 1].max() / 1e3), (0.05, arr[:, 2].max() / 1e3))


def main():
    stas = np.load(HERE / "code_inputs1" / "stas_xyz.npy").astype(float)
    d_obs = np.load(HERE / "code_inputs1" / "tobs_p.npy").ravel()
    vel_model = loc.load_velocity_model(str(MODEL), grid_step_km=(0.02, 0.02, 0.02), pad_km=0.3,
                                        verbose=False)
    fields = loc.StationTravelTimeFields(vel_model, stas, verbose=False)
    survey = survey_extent()

    print("=== Test 1: travel-time derivatives ===")
    p = np.array([392.70, 4865.75, 0.25])
    G = fields.derivatives(p)
    Gfd = np.zeros_like(G)
    h = 0.01
    for j in range(3):
        a, b = p.copy(), p.copy()
        a[j] += h
        b[j] -= h
        Gfd[:, j] = (fields.travel_times(a) - fields.travel_times(b)) / (2 * h)
    print(f"  max |analytic - finite difference| = {np.abs(G - Gfd).max():.2e} s/km")
    print(f"  rank of the station Jacobian = {np.linalg.matrix_rank(G)} (3 is required)")

    print(f"\n=== Test 2: recovery of synthetic sources ({PICK_NOISE_S} s pick noise) ===")
    rng = np.random.default_rng(SEED)
    eh, ev = [], []
    for true in SYNTHETIC_SOURCES:
        t = fields.travel_times(np.array(true)) + 0.5 + rng.normal(0, PICK_NOISE_S, len(stas))
        hyc, cov, res = loc.locate(fields, t, search_extent_km=survey, verbose=False)
        dh = 1000 * float(np.linalg.norm(hyc[:2] - true[:2]))
        dz = 1000 * float(hyc[2] - true[2])
        eh.append(dh)
        ev.append(abs(dz))
        print(f"  true {true} -> {np.round(hyc[:3], 4).tolist()}  horizontal {dh:5.1f} m  "
              f"depth {dz:+6.1f} m  origin time {hyc[3] - 0.5:+.3f} s  RMS {np.sqrt((res ** 2).mean()):.4f} s")
    print(f"  horizontal error: median {np.median(eh):.0f} m, maximum {np.max(eh):.0f} m")
    print(f"  depth error:      median {np.median(ev):.0f} m, maximum {np.max(ev):.0f} m")

    print("\n=== Test 3: released picks in code_inputs1 ===")
    hyc, cov, res = loc.locate(fields, d_obs, search_extent_km=survey, verbose=False)
    loc.present_loc_results(hyc, cov, res)
    print("  residuals (s):", np.round(res, 4).tolist())


if __name__ == "__main__":
    main()
