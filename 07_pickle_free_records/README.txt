Pickle-free classified event records

This folder contains lossless NPZ copies of all 17,622 classified event records
in the revised dataset. It serves the verification of the dataset in the same
way as the code package and is not part of the dataset release. Because of its
data volume it is not stored in the GitHub code repository but deposited
separately on Figshare: https://doi.org/10.6084/m9.figshare.34128987
It can be moved and distributed as one folder. The original NPY files remain
in the dataset (https://doi.org/10.6084/m9.figshare.34074750).

Contents

classified event-level waveform data (npz)/
    17,622 NPZ files in five class folders:
      blasting              6,512
      mechanical_mining     7,690
      microseismic          1,401
      small_landslide          148
      transport_vehicles    1,871
    Each file preserves the source NPY filename stem and all six field values.
records_index.csv
    One UTF-8 CSV index with 17,622 records and a header row. It lists class,
    both file paths, sensor ID, trigger start/end, sampling rate, sample count,
    and separate SHA-256 hashes for the NPY and NPZ files.
load_event.py
    Loads the indexed NPZ by default with allow_pickle=False and checks its
    SHA-256, schema, sampling interval and metadata. Also verifies all records;
    the optional --report PATH option writes the result to a JSON file.
build_npz_release.py
    Recreates the NPZ release and index from the trusted original NPY dataset.
    Each written NPZ is reopened with allow_pickle=False and every field is
    compared exactly against its source. Existing release files are not replaced.
conversion_report.json
    Counts, field-comparison results and conversion provenance.

Record format

sensor_id       Scalar Unicode array; original station identifier.
event_start     Scalar Unicode array; original STA/LTA trigger onset in UTC.
event_end       Scalar Unicode array; original STA/LTA trigger termination in UTC.
time_axis       One-dimensional float64 array, seconds from the first sample.
signal          One-dimensional float64 array; original waveform values unchanged.
sampling_rate   Scalar float64 array, Hz; the stored value is preserved exactly.

No filtering, resampling, amplitude scaling, time adjustment or relabelling is
performed during conversion. The two formats contain equal field values but
have different byte representations and therefore different SHA-256 hashes.
NPZ files contain standard NPY arrays in a ZIP container, with no object arrays
or pickle payload. Other languages require a compatible NPY/NPZ reader with
support for NumPy Unicode arrays; native support is not implied.

Index paths

record_id is class/file_stem and uniquely identifies a record.
npz_path is relative to this folder (the folder containing records_index.csv).
npy_path is relative to the original dataset root passed as --dataset-root.
Paths use forward slashes and do not contain machine-specific absolute paths.
Both timestamps and sensor IDs should be read as text; do not let spreadsheet
software rewrite them. The original sampling-rate precision is retained.

Quick start

Python 3.11+ and NumPy are sufficient for this folder. Run these commands from
the parent code-dataset folder; quote paths that contain spaces.

1. Verify every bundled NPZ file without the original dataset:

python 07_pickle_free_records/load_event.py --verify-all

2. Load one record by copying its record_id from records_index.csv:

python 07_pickle_free_records/load_event.py "blasting/sensor_230000081_20250603T103254.752000_20250603T103257.320000"

3. Also verify the SHA-256 hashes of all original NPY files:

python 07_pickle_free_records/load_event.py --verify-all --include-npy --dataset-root "../dataset"

This command hashes the NPY files without unpickling them.

4. Explicitly load the original NPY when required:

python 07_pickle_free_records/load_event.py "blasting/sensor_230000081_20250603T103254.752000_20250603T103257.320000" --npy --trust-source --dataset-root "../dataset"

An explicit trust flag and a matching indexed hash are both required for NPY.
A checksum detects changes relative to the index; it does not authenticate the
publisher or make an untrusted pickle safe. Obtain the index from the same
trusted release as the data.

Python use

From this folder:

from load_event import load_event
event = load_event("blasting/sensor_230000081_20250603T103254.752000_20250603T103257.320000")
signal = event["signal"]
time = event["time_axis"]

The returned dictionary contains the six documented fields. The loader resolves
its default index relative to its own file, independent of the working directory.
For another extracted copy, pass index_path="/path/to/records_index.csv".
With NumPy directly, use np.load(path, allow_pickle=False); direct NumPy loading
does not check SHA-256, so use load_event.py when integrity verification is needed.

Rebuild

python 07_pickle_free_records/build_npz_release.py --dataset-root "../dataset" --output-root "../rebuilt_npz_release" --trust-source

The output directory must not already contain an NPZ data folder, record index
or conversion report. Rebuilding only writes new files; the source is read-only.
Compressed bytes can vary between software versions; use the newly generated
index for a rebuilt release. Scientific field values are checked for exact equality.

Relationship to the rest of the code

04_data_loading and the existing classification workflow retain their original
NPY interface. This folder provides an additional portable verification copy
and loader; it does not switch training scripts to NPZ automatically. To
distribute the complete verification copy, include this whole folder or its
matching ZIP, not only the two Python scripts. The original NPY dataset is optional for NPZ
loading and required only for NPY verification, legacy loading, or rebuilding.
