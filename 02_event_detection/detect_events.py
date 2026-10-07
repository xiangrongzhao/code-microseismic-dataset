"""STA/LTA candidates and overlapping trigger groups; expert labels are not recreated."""
from pathlib import Path
import argparse,json,csv
import numpy as np
from obspy import UTCDateTime
from obspy.signal.trigger import classic_sta_lta,trigger_onset
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / '01_preprocessing'))
from preprocess_raw import preprocess

def detect_file(path):
    tr=preprocess(path);fs=float(tr.stats.sampling_rate);data=np.asarray(tr.data)
    cft=classic_sta_lta(data,int(.5*fs),int(3*fs));events=[]
    for on,off in trigger_onset(cft,3.5,1.5):
        lo=max(0,int(on)-int(2*fs));hi=min(len(data)-1,int(off)+int(8*fs))
        events.append({'sensor_id':Path(path).name.split('.')[0],
          'event_start':(tr.stats.starttime+on/fs).isoformat(),
          'event_end':(tr.stats.starttime+off/fs).isoformat(),
          'full_signal':data[lo:hi+1],
          'full_time_axis':np.arange(lo,hi+1)/fs-on/fs,
          'amplitude_mean':float(np.abs(data[on:off+1]).mean())})
    return events

def group_candidates(events,minimum_stations=3):
    events=sorted(events,key=lambda x:UTCDateTime(x['event_start']));used=set();groups=[]
    for i,anchor in enumerate(events):
        if i in used:continue
        start,end=UTCDateTime(anchor['event_start']),UTCDateTime(anchor['event_end'])
        selected=[i]
        for j in range(i+1,len(events)):
            if UTCDateTime(events[j]['event_start'])>end:break
            if j not in used and events[j]['amplitude_mean']>0 and UTCDateTime(events[j]['event_end'])>=start:selected.append(j)
        # Correct the old script's record-count test: count distinct stations.
        if len({events[j]['sensor_id'] for j in selected})>=minimum_stations:
            groups.append([events[j] for j in selected]);used.update(selected)
    return groups

def extract_record(event):
    x=np.asarray(event['full_signal']);t=np.asarray(event['full_time_axis'])
    fs=1/(t[1]-t[0]) if len(t)>1 else 250.
    lo=max(0,int((-2-t[0])*fs));hi=min(len(x),int((8-t[0])*fs))
    if hi<=lo:raise ValueError('Empty extraction window.')
    return {'sensor_id':event['sensor_id'],'event_start':event['event_start'],'event_end':event['event_end'],
      'time_axis':t[lo:hi]-t[lo],'signal':x[lo:hi],'sampling_rate':fs}

def main():
    a=argparse.ArgumentParser(description=__doc__);a.add_argument('--raw-dir',type=Path,required=True)
    a.add_argument('--out-dir',type=Path,required=True);a.add_argument('--limit',type=int)
    args=a.parse_args();raw=args.raw_dir.resolve();out=args.out_dir.resolve()
    released=raw.parent
    if raw==out or raw in out.parents or (released/'classified event-level waveform data') in out.parents or out==released/'classified event-level waveform data':
        raise ValueError('Candidate outputs must be separate from released waveforms.')
    out.mkdir(parents=True,exist_ok=True);files=sorted(raw.rglob('*.DHZ.miniseed'))
    if args.limit:files=files[:args.limit]
    events=[];errors=[]
    for p in files:
        try:events.extend(detect_file(p))
        except ValueError as e:errors.append({'file':p.name,'reason':str(e)})
    groups=group_candidates(events);written=set()
    for g in groups:
        for event in g:
            r=extract_record(event)
            stamp=lambda s:UTCDateTime(s).strftime('%Y%m%dT%H%M%S.%f')
            name=f"sensor_{r['sensor_id']}_{stamp(r['event_start'])}_{stamp(r['event_end'])}.npy"
            if name not in written:np.save(out/name,r);written.add(name)
    report={'raw_files':len(files),'trigger_records':len(events),'overlap_groups':len(groups),'single_station_candidates':len(written),
      'group_rule':'overlapping trigger intervals, at least three distinct stations','expert_annotation_reproduced':False,'skipped_files':errors}
    (out/'detection_summary.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report))
if __name__=='__main__':main()
