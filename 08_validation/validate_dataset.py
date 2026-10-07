"""Check event schema, filenames, labels and identical payloads from the actual inputs."""
from pathlib import Path
import argparse, collections, hashlib, json, re, sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'06_data_loading'))
from load_event import load_event

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--events',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--trust-source',action='store_true',required=True);args=p.parse_args()
    counts=collections.Counter();digests=collections.defaultdict(list);lengths=[];failures=[]
    pattern=re.compile(r'^sensor_[0-9]{9}_[0-9]{8}T[0-9]{6}\.[0-9]{6}_[0-9]{8}T[0-9]{6}\.[0-9]{6}\.npy$')
    files=sorted(args.events.rglob('*.npy'))
    for i,f in enumerate(files):
        try:
            if not pattern.fullmatch(f.name):raise ValueError('Nonstandard filename')
            _,data=load_event(f,trusted=args.trust_source)
            if f.name.split('_')[1]!=str(data['sensor_id']):raise ValueError('Filename and stored sensor ID disagree')
            counts[f.parent.name]+=1;lengths.append(len(data['signal']))
            digests[hashlib.sha256(f.read_bytes()).hexdigest()].append(f.relative_to(args.events).as_posix())
        except Exception as e:failures.append({'file':str(f),'error':str(e)})
        if (i+1)%2000==0:print(f'Validated {i+1}/{len(files)}',flush=True)
    cross=[v for v in digests.values() if len({Path(n).parent.name for n in v})>1]
    report={'files':len(files),'class_counts':dict(counts),'sample_length_min':min(lengths) if lengths else None,'sample_length_max':max(lengths) if lengths else None,'cross_label_identical_groups':len(cross),'cross_label_identical_records':sum(map(len,cross)),'cross_label_files':cross,'failures':failures}
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='cross_label_files'},indent=2))
    if failures:raise SystemExit(1)
if __name__=='__main__':main()
