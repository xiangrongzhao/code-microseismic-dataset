from pathlib import Path
import csv,json,datetime,collections,re,hashlib,sys,zipfile
import numpy as np
from obspy import read
DATA=Path(__file__).resolve().parent.parent;WORK=None
DAY=datetime.datetime(2025,6,1,tzinfo=datetime.timezone.utc).timestamp();END=DAY+86400
CHS=['DHE','DHN','DHZ']
def csvread(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def csvwrite(p,rows):
    with p.open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def iso(t):return datetime.datetime.fromtimestamp(t,datetime.timezone.utc).isoformat(timespec='microseconds')
def merge(items):
    out=[]
    for s,e in sorted(items):
        s=max(s,DAY);e=min(e,END)
        if e<=s:continue
        if out and s<=out[-1][1]+0.000001:out[-1][1]=max(out[-1][1],e)
        else:out.append([s,e])
    return out
def intersect(a,b):
    out=[];i=j=0
    while i<len(a) and j<len(b):
        s=max(a[i][0],b[j][0]);e=min(a[i][1],b[j][1])
        if e>s:out.append([s,e])
        if a[i][1]<b[j][1]:i+=1
        else:j+=1
    return out
def duration(items,s=DAY,e=END):return sum(max(0,min(e,y)-max(s,x)) for x,y in items)
def main():
    global DATA,WORK
    import argparse
    a=argparse.ArgumentParser(description='Decode actual sample intervals and export June 1 coverage tables without writing source waveforms.')
    a.add_argument('--dataset',type=Path,default=Path(__file__).resolve().parents[2]/'dataset');a.add_argument('--limit',type=int);a.add_argument('--out-dir',type=Path,required=True);args=a.parse_args()
    DATA=args.dataset;WORK=args.out_dir;WORK.mkdir(parents=True,exist_ok=True)
    if WORK.resolve()==DATA.resolve():a.error('Use a separate output directory.')
    ids=set()
    for book in (DATA/'station metadata').glob('*.xlsx'):
        with zipfile.ZipFile(book) as z:
            for n in z.namelist():
                if n.startswith('xl/') and n.endswith('.xml'):
                    ids.update(re.findall(r'(?<![0-9])230000[0-9]{3}(?![0-9])',z.read(n).decode('utf-8')))
    files=sorted((DATA/'raw waveform data').rglob('*.miniseed'))
    if args.limit:files=files[:args.limit]
    rows=[]
    for p in files:
        pieces=p.name.split('.');sid=pieces[0];component=pieces[-2]
        if component not in CHS:raise ValueError('Unexpected filename component: '+p.name)
        ids.add(sid)
        rows.append({'file':p.relative_to(DATA).as_posix(),'sensor_id':sid,'filename_component':component,'bytes':p.stat().st_size,'sampling_rate_hz':250.})
    ids=sorted(ids)
    if not rows:raise ValueError('No raw miniSEED files found')
    intervals=collections.defaultdict(list);records=[];filegroups=collections.defaultdict(dict);perfile=[]
    for i,r in enumerate(rows):
        p=DATA/r['file'];st=read(str(p),format='MSEED')
        assert len(st)>0
        spans=[];npts=0;gaps=st.get_gaps();channels=set();rates=set();nslcs=set()
        for t in st:
            assert len(t.data)==t.stats.npts and np.isfinite(t.data).all()
            fs=float(t.stats.sampling_rate);s=float(t.stats.starttime);e=s+t.stats.npts/fs
            spans.append([s,e]);npts+=int(t.stats.npts);channels.add(t.stats.channel);rates.add(fs);nslcs.add(t.id)
            intervals[(r['sensor_id'],r['filename_component'])].append([s,e])
            records.append({'file':r['file'],'sensor_id':r['sensor_id'],'filename_component':r['filename_component'],'header_nslc':t.id,'trace_start_utc':iso(s),'trace_end_exclusive_utc':iso(e),'sample_count':int(t.stats.npts),'sampling_rate_hz':fs})
        assert len(rates)==1 and list(rates)[0]==250
        rr=dict(r);rr['bytes']=int(r['bytes']);rr['sampling_rate_hz']=float(r['sampling_rate_hz']);rr.update({'first_sample_utc':iso(min(s for s,e in spans)),'end_exclusive_utc':iso(max(e for s,e in spans)),'header_nslc':';'.join(sorted(nslcs)),'trace_count':len(st),'sample_count':npts,'internal_gap_or_overlap_count':len(gaps),'decoded_finite':1})
        perfile.append(rr)
        key=Path(r['file']).name.rsplit('.',2)[0];filegroups[key][r['filename_component']]=rr
        if i%3000==0:print('Decoded raw files: %d/%d'%(i,len(rows)),flush=True)
    merged={(sid,c):merge(intervals[(sid,c)]) for sid in ids for c in CHS}
    common={sid:intersect(intersect(merged[(sid,'DHE')],merged[(sid,'DHN')]),merged[(sid,'DHZ')]) for sid in ids}
    grids=[];gaps=[];station=[];chan=[];triples=[]
    for key,parts in sorted(filegroups.items()):
        r=next(iter(parts.values()))
        triples.append({'station_time_key':key,'sensor_id':r['sensor_id'],'start_header':r['first_sample_utc'],'end_exclusive_header':r['end_exclusive_utc'],'DHE_available':int('DHE' in parts),'DHN_available':int('DHN' in parts),'DHZ_available':int('DHZ' in parts),'complete_triplet':int(len(parts)==3)})
    bysid=collections.Counter(r['sensor_id'] for r in rows)
    bychan=collections.Counter((r['sensor_id'],r['filename_component']) for r in rows)
    for sid in ids:
        seconds=[duration(merged[(sid,c)]) for c in CHS]
        cs=duration(common[sid])
        station.append({'sensor_id':sid,'file_count':bysid[sid],'DHE_seconds':round(seconds[0],6),'DHN_seconds':round(seconds[1],6),'DHZ_seconds':round(seconds[2],6),'common_three_component_seconds':round(cs,6),'common_day_fraction':cs/86400,'present_in_raw_release':int(bysid[sid]>0)})
        for c in CHS:
            seq=merged[(sid,c)];total=duration(seq)
            chan.append({'sensor_id':sid,'filename_component':c,'header_channel':{'DHE':'DHN','DHN':'DHE','DHZ':'DHZ'}[c],'file_count':bychan[(sid,c)],'covered_seconds_actual_traces':round(total,6),'day_fraction':total/86400,'first_sample_header':iso(seq[0][0]) if seq else '', 'end_exclusive_header':iso(seq[-1][1]) if seq else ''})
            cur=DAY
            for s,e in seq:
                if s>cur:gaps.append({'sensor_id':sid,'filename_component':c,'gap_start_utc':iso(cur),'gap_end_exclusive_utc':iso(s),'gap_seconds':round(s-cur,6)})
                cur=max(cur,e)
            if cur<END:gaps.append({'sensor_id':sid,'filename_component':c,'gap_start_utc':iso(cur),'gap_end_exclusive_utc':iso(END),'gap_seconds':round(END-cur,6)})
        for n in range(288):
            s=DAY+n*300;e=s+300;ds=[duration(merged[(sid,c)],s,e) for c in CHS];cc=duration(common[sid],s,e)
            status='complete_3C' if cc>=300-0.000001 else 'partial_3C' if cc>0 else 'partial_components' if any(ds) else 'absent'
            grids.append({'sensor_id':sid,'interval_start_utc':iso(s),'interval_end_exclusive_utc':iso(e),'DHE_covered_s':round(ds[0],6),'DHN_covered_s':round(ds[1],6),'DHZ_covered_s':round(ds[2],6),'common_3C_covered_s':round(cc,6),'status':status})
    csvwrite(WORK/'raw_file_inventory.csv',perfile);csvwrite(WORK/'raw_trace_intervals.csv',records)
    csvwrite(WORK/'raw_triplet_availability.csv',triples);csvwrite(WORK/'raw_channel_availability.csv',chan)
    csvwrite(WORK/'raw_header_envelope_gaps.csv',gaps);csvwrite(WORK/'raw_five_minute_coverage.csv',grids)
    csvwrite(WORK/'raw_station_coverage.csv',station)
    report={'files':len(rows),'registered_stations':len(ids),'released_stations':len(bysid),'trace_records':len(records),'decoded_finite_files':sum(r['decoded_finite'] for r in perfile),'files_with_internal_gap_or_overlap':sum(r['internal_gap_or_overlap_count']>0 for r in perfile),'station_time_groups':len(triples),'complete_triplets':sum(r['complete_triplet'] for r in triples),'incomplete_triplets':[r for r in triples if not r['complete_triplet']],'registered_stations_absent':[s for s in ids if not bysid[s]],'coverage_grid_rows':len(grids),'grid_status_counts':dict(collections.Counter(r['status'] for r in grids)),'missing_interval_rows':len(gaps),'limited_run':args.limit is not None,'reference_timezone':'UTC','end_interval_definition':'last_sample + 1/sampling_rate'}
    (WORK/'coverage_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf-8')
    sheets={'station':station,'files':perfile,'triplets':triples,'grid':grids,'gaps':gaps,'report':report}
    (WORK/'coverage_workbook_data.json').write_text(json.dumps(sheets,ensure_ascii=False),encoding='utf-8')
    print(json.dumps(report,ensure_ascii=True),flush=True)
if __name__=='__main__':main()
