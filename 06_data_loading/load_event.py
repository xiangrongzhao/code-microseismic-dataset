"""Load documented event dictionaries after explicit trust, with optional hash verification."""
from pathlib import Path
import hashlib
import numpy as np

SINGLE = {"sensor_id", "event_start", "event_end", "time_axis", "signal", "sampling_rate"}

def load_event(path, expected_sha256=None, trusted=False):
    """A matching hash checks integrity, not whether pickle code is trustworthy."""
    if not trusted:
        raise ValueError("Only load pickle-bearing files from a source you explicitly trust.")
    p=Path(path)
    if expected_sha256 is not None and hashlib.sha256(p.read_bytes()).hexdigest() != expected_sha256.lower():
        raise ValueError("File SHA-256 does not match the expected value.")
    value=np.load(str(p),allow_pickle=True).item()
    if not isinstance(value,dict):
        raise ValueError("Expected a serialized dictionary.")
    keys=set(value)
    if keys==SINGLE:
        arrays=[np.asarray(value["signal"])]
        schema="single_station"
    else:
        raise ValueError("Unknown event schema.")
    t=np.asarray(value["time_axis"])
    fs=float(value["sampling_rate"])
    if t.ndim!=1 or not np.isfinite(t).all() or not np.isfinite(fs) or fs<=0:
        raise ValueError("Invalid time axis or sampling rate.")
    for a in arrays:
        if a.ndim!=1 or len(a)!=len(t) or not np.isfinite(a).all():
            raise ValueError("Waveform and time-axis structure mismatch.")
    timing_atol = 1e-10
    if len(t)>1 and not np.allclose(np.diff(t),1/fs,rtol=1e-6,atol=timing_atol):
        raise ValueError("Time-axis spacing does not match sampling rate.")
    return schema,value

if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file")
    parser.add_argument("--sha256", help="Optional externally supplied SHA-256")
    parser.add_argument("--trust-source",action="store_true")
    args=parser.parse_args()
    schema,value=load_event(args.file,args.sha256,args.trust_source)
    print("schema:",schema,"fields:",", ".join(sorted(value)))
    print("samples:",len(value["time_axis"]),"sampling rate (Hz):",value["sampling_rate"])
