"""Screen and band-pass raw vertical records without changing source files."""
from pathlib import Path
import numpy as np
from obspy import read

def preprocess(path,minimum_samples=70000):
    path=Path(path)
    if not path.name.endswith('.DHZ.miniseed'):
        raise ValueError('This extraction example requires a filename-DHZ record.')
    st=read(str(path),format='MSEED')
    if len(st)!=1:
        raise ValueError('Multiple traces or internal gaps require separate handling.')
    trace=st[0]
    if trace.stats.npts<=minimum_samples:
        raise ValueError('Record does not exceed the 70000-sample screening threshold.')
    if not np.isfinite(trace.data).all():raise ValueError('Nonfinite raw samples.')
    # Match the archived extraction source: no added detrend or response removal.
    trace.filter('bandpass',freqmin=1.0,freqmax=20.0,corners=4,zerophase=True)
    return trace

if __name__=='__main__':
    import argparse,json
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('files',type=Path,nargs='*');a.add_argument('--out-dir',type=Path,required=True)
    a.add_argument('--raw-dir',type=Path);a.add_argument('--limit',type=int)
    args=a.parse_args();args.out_dir.mkdir(parents=True,exist_ok=True);rows=[]
    files=args.files+sorted(args.raw_dir.rglob('*.DHZ.miniseed')) if args.raw_dir else args.files
    if args.limit:files=files[:args.limit]
    if not files:a.error('Supply raw files or --raw-dir.')
    for p in files:
        try:tr=preprocess(p)
        except ValueError as e:
            rows.append({'source':p.name,'skipped':str(e)});continue
        out=args.out_dir/(p.name+'.filtered.npz')
        if out.resolve()==p.resolve():raise ValueError('Source cannot be an output.')
        np.savez_compressed(out,signal=tr.data,sampling_rate=float(tr.stats.sampling_rate),start_utc=str(tr.stats.starttime))
        rows.append({'source':p.name,'output':out.name,'samples':int(tr.stats.npts),'sampling_rate_hz':float(tr.stats.sampling_rate)})
    (args.out_dir/'preprocessing.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
