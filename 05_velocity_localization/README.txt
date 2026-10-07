# Event localization example

This package reproduces the localization example of the paper end to end, from the waveform records
to the numbers and the panels (c) and (d) of Figure 5 reported in the Methods section. It is
self-contained: the seven waveform records of the example event, the station table and the velocity
model are included, so no other download is needed.

The example event is the record group labelled as a small landslide on 9 June 2025 at 22:01:52 UTC,
recorded by seven southern-highwall stations. Its source lies inside the station array and inside
the area covered by the released velocity survey, and the wavefront curvature across the array is
measurable: a point source fits the arrival times with a residual of 2 ms against 53 ms for a plane
wave, which is the limiting case of a distant source.

## Contents

| Path | Description |
|---|---|
| `waveforms/` | The seven classified records of the event (`.npy`), copied unchanged from the small-landslide category of the dataset. |
| `stations.csv` | The 36 stations of Table 2 of the paper: array, WGS84 coordinates (DMS), elevation, and x, y, z in the frame of the velocity model (km). |
| `XJD2_updated.txt` | The released velocity model: X, Y, Z in metres and V<sub>S</sub> in km/s. |
| `make_inputs1.py` | Picks the P-wave onsets automatically and writes `code_inputs1/`. |
| `loc_xjd2.py` | Velocity-model loading, eikonal travel-time fields, grid search and Levenberg-Marquardt inversion. |
| `run_localization_example.py` | Computes both solutions, the station jackknife, and panels (c) and (d) of Figure 5. |
| `test_loc_xjd2.py` | Verifies the travel-time derivatives and the rank of the Jacobian, and recovers synthetic sources of known position. |
| `code_inputs1/` | `stas_xyz.npy`, `tobs_p.npy` and `pick_report.csv` for the example event (written by `make_inputs1.py`), and `input_provenance.json` with the SHA-256 hashes of the records, the station table and these three files. |
| `localization_outputs/` | `localization_summary.json`, the catalogue file `example_catalogue.csv`, and the figure panels (written by `run_localization_example.py`). |
| `plot_velocity_slices.py` | Optional V<sub>S</sub> sections of `XJD2_updated.txt` (`python plot_velocity_slices.py --out <directory>`). |
| `requirements.txt` | Package versions used to produce the published numbers (Python 3.11). |

Each `.npy` record stores a Python dictionary with the keys `sensor_id`, `event_start`, `event_end`,
`time_axis`, `signal` and `sampling_rate` (vertical component, 250 Hz, 10 s starting 2 s before the
trigger onset). Load it with `numpy.load(path, allow_pickle=True).item()`.

## Usage

```
pip install -r requirements.txt
python make_inputs1.py
python run_localization_example.py
python test_loc_xjd2.py
```

`make_inputs1.py` reads `waveforms/` by default. To start from a downloaded copy of the dataset
instead, pass its small-landslide directory as the first argument; the records of the event are
selected by station and trigger time. Each script runs in a few seconds on a desktop computer.
Rerunning the three scripts reproduces `localization_summary.json` and `example_catalogue.csv` exactly.

## Method

1. Each record is band-pass filtered (2–20 Hz, fourth-order Butterworth, zero phase), and the P onset
   is picked with the Akaike information criterion between 1.0 and 3.5 s of the record; picks with a
   signal-to-noise ratio below 3 are discarded.
2. V<sub>S</sub> from the velocity model is converted to V<sub>P</sub> with a Poisson ratio of 0.25
   and interpolated onto a 0.02 km grid.
3. Travel times are computed with the fast marching method. By reciprocity, the travel time from a
   trial source to a station equals the travel time computed with that station as the point source,
   so each station has its own travel-time field and its own row of the Jacobian.
4. The initial location follows the minimum travel-time principle, evaluated as a grid search over
   the velocity grid with the origin time solved analytically, and is refined by Levenberg-Marquardt
   inversion of (x, y, z, t<sub>0</sub>). The search is restricted to the volume covered by the
   velocity survey.
5. A second solution inverts a single homogeneous P velocity jointly with the source,
   (x, y, z, V<sub>P</sub>, t<sub>0</sub>).
6. Removing each station in turn (jackknife) gives the sensitivity of each solution to the station
   set.

## Coordinates

Station coordinates are given in the CGCS2000 three-degree Gauss-Krüger projection on the 96°E
central meridian (EPSG:4541), shifted by a fixed offset of −1.79 m in easting and +85.86 m in
northing, and expressed in kilometres. This is the frame in which `XJD2_updated.txt` is defined
once its 32,000,000 m offset is removed from the X column. Elevation is positive upwards, in
kilometres, for both the stations and the velocity model.

## Results

| | Released velocity model | Homogeneous velocity |
|---|---|---|
| x, y, z (km) | 392.275, 4865.942, 0.159 | 392.495, 4865.971, 0.216 |
| V<sub>P</sub> (km/s) | from the model, 0.173–1.732 | 0.491, inverted with the source |
| Origin time (s) | −0.3975 | −0.2400 |
| Travel-time residual (s, RMS) | 0.057 | 0.002 |
| Jackknife, largest shift (m) | 0 | 35 |

`example_catalogue.csv` lists both solutions in catalogue form:

| | Released velocity model | Homogeneous velocity |
|---|---|---|
| Origin time (UTC) | 2025-06-09 22:01:51.366 | 2025-06-09 22:01:51.524 |
| Stations / picks | 7 / 7 | 7 / 7 |
| Azimuthal gap (°) | 309 | 176 |
| Minimum epicentral distance (m) | 156 | 45 |
| Horizontal / vertical error (m) | 128 / 150 (formal 1σ) | 8 / 34 (jackknife, largest shift) |
| RMS (s) | 0.057 | 0.002 |

The two solutions are complementary. The arrival times are reproduced by a constant P velocity of
about 0.49 km/s, which is characteristic of the loose, unconsolidated waste material placed during
open-pit stripping. Four of the seven stations stand on this material between 351 and 373 m
elevation, above the 338 m upper surface of the ambient-noise model, so the homogeneous solution
represents the loose layer directly beneath the stations and the model-based solution the deeper
structure resolved by the survey. The 229 m separation between them shows how the near-surface
velocity influences location estimates at this site.

## Verification

`test_loc_xjd2.py` checks the inversion itself rather than the result for any one event.

1. The analytic derivatives of travel time with respect to source position agree with finite
   differences of the computed travel times to within 10<sup>−10</sup> s/km, and the station
   Jacobian has full rank (3).
2. Five synthetic sources inside the array, whose positions are known by construction, are recovered
   from travel times perturbed by 0.01 s of pick noise. With the seven-station geometry of the
   example, the horizontal error has a median of 64 m and a maximum of 94 m, and the depth error a
   median of 19 m and a maximum of 169 m; depth is less tightly constrained than horizontal position,
   as expected for an array deployed on the surface.
