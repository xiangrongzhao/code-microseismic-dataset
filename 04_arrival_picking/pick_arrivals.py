"""Vertical-component AR picking example. Does not supply a 158-event catalogue.
Original three-component AR picker is called with zero horizontal inputs and S picking disabled.
"""
import numpy as np
from obspy.signal.filter import bandpass
from obspy.signal.trigger import ar_pick
def pick_first_break(
        z_data: np.ndarray,  # Vertical-component samples only.
        samp_rate: float,
        f1: float = 1.0,
        f2: float = 20.0,
        lta_p: float = 1.0,
        sta_p: float = 0.1,
        min_pick: float = 0.05
) -> float:
    """
    Pick the P-wave first arrival from a single vertical component.
    Args:
        z_data: Vertical-component waveform array.
        samp_rate: Sampling rate in Hz.
        f1/f2: Lower and upper band-pass frequencies in Hz.
        lta_p/sta_p: P-wave LTA and STA windows in seconds.
        min_pick: Minimum accepted arrival time in seconds.
    Returns:
        P-wave arrival in seconds relative to the first sample, or np.nan.
    """
    # 1. Check that samples are available.
    if len(z_data) == 0:
        return np.nan

    # 2. Remove the mean and apply band-pass filtering.
    z = z_data - np.mean(z_data)
    z = bandpass(z, f1, f2, samp_rate, corners=4, zerophase=True)

    # 3. Check the minimum waveform length required by the picker.
    n_min = int((lta_p + sta_p) * samp_rate) + 10 + 1
    if len(z) < n_min:
        return np.nan

    # 4. Check that the signal has a finite, nonzero amplitude.
    data_max = np.max(np.abs(z))
    if data_max == 0 or not np.isfinite(data_max):
        return np.nan

    # 5. Run the AR P-wave picker with the vertical component.
    # ar_pick requires three inputs; supply zeros for the N and E components.
    n = np.zeros_like(z)
    e = np.zeros_like(z)
    try:
        p_sec, _ = ar_pick(
            z, n, e, samp_rate,
            f1, f2,
            lta_p, sta_p, 1.0, 0.1,  # Placeholder S-wave parameters.
            10, 10, 0.5, 0.5,
            s_pick=False  # Disable S-wave picking.
        )
        # Reject arrivals earlier than the acceptance threshold.
        if p_sec < min_pick:
            return np.nan
        return p_sec
    except Exception as e:
        print(f"P-wave picking failed: {e}")
        return np.nan

if __name__ == "__main__":
    import argparse,json,sys
    from pathlib import Path
    sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06_data_loading'))
    from load_event import load_event
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--events',type=Path,nargs='+',required=True)
    a.add_argument('--out',type=Path,required=True)
    a.add_argument('--trust-source',action='store_true',required=True)
    args=a.parse_args();rows=[]
    for p in args.events:
        _,event=load_event(p,trusted=args.trust_source)
        sec=pick_first_break(np.asarray(event['signal']),float(event['sampling_rate']))
        rows.append({'file':p.name,'sensor_id':event['sensor_id'],'relative_pick_s':float(sec) if np.isfinite(sec) else None,'reference':'first stored sample; not the trigger onset','absolute_time_available':False})
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(rows,indent=2),encoding='utf-8')
