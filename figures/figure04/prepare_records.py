"""Verify the source traces, identify a single landslide record and rebuild the processed NPZs.
Requires NumPy, SciPy, ObsPy and PyMuPDF. Run from an unpacked copy with:
    python prepare_records.py
    python plot_figure4.py
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import tempfile

import fitz
import numpy as np
from obspy import UTCDateTime
from obspy.signal.filter import bandpass

from preprocess_raw import preprocess

HERE = Path(__file__).resolve().parent
SRC = HERE / "sources"
FS = 250.
# Each PDF start is the recovered original 10 s plotting window (inclusive endpoints).
# The plotted NPY is the processed event-centred archive window and retains its own start.
MINE = [
    ("a","Microseismic","microseismic","230000084","20250609T183326","2025-06-09T18:33:26.000000","微震示例图.pdf"),
    ("b","Blasting","blasting","230000081","20250604T104333","2025-06-04T10:43:32.000000","爆破示例图.pdf"),
    ("d","Mechanical mining","mechanical_mining","230000120","20250609T165142","2025-06-09T16:51:42.000000","机械开采示例图.pdf"),
    ("e","Transport vehicles","transport_vehicles","230000107","20250607T215253","2025-06-07T21:52:53.000000","运输车辆示例图.pdf"),
]

def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def guide_curve(pdf,n):
    with fitz.open(str(pdf)) as doc:
        paths=[d for d in doc[0].get_drawings() if d['color'] and
               np.allclose(d['color'],(0,.447,.741),atol=.006) and len(d['items'])>300]
        if len(paths)!=1:
            raise ValueError('Expected one blue waveform in '+pdf.name)
        items=paths[0]['items']
        pts=[items[0][1]]+[item[2] for item in items]
    xx=np.array([pt.x for pt in pts]); yy=-np.array([pt.y for pt in pts])
    return np.interp(np.linspace(xx[0],xx[-1],n),xx,yy)

def corr(x,y):
    return float(np.corrcoef(x,y)[0,1])

def one(pattern):
    a=list(SRC.glob(pattern))
    if len(a)!=1: raise ValueError('Expected one file for '+pattern)
    return a[0]

def save_record(dest,row,x,t,raw=None):
    native=np.asarray(x,dtype=np.float64)
    t=np.asarray(t,dtype=np.float64)
    assert native.shape==t.shape and native.ndim==1 and np.isfinite(native).all()
    assert np.allclose(np.diff(t),1/FS,rtol=0,atol=1e-9)
    if not np.max(np.abs(native))>0: raise ValueError('Empty waveform')
    np.savez_compressed(str(dest/(row['panel']+'.npz')),signal=native,time_axis=t,
        sampling_rate=np.float64(FS),source_id=row['identifier'],unit=row['native_unit'],
        component=row['component'],trigger_utc=row['trigger_utc'],category=row['label'])
    row.update(n=len(native),native_peak_abs=float(np.max(np.abs(native))),
               signal_sha256=hashlib.sha256(native.astype('<f8').tobytes()).hexdigest())
    return row

def base_row(panel,label):
    keys=('panel','label','identifier','trigger_utc','record_start_utc','component','native_unit',
          'source_status','guide_pdf','guide_start_utc','guide_correlation','release_relative_path',
          'original_source','original_sha256','processing_input','processing_input_sha256',
          'sample_offset','raw_filter_max_abs_error','matlab_record','matlab_channel','filtered_template_correlation','guide_match_score','guide_envelope_cosine')
    row=dict.fromkeys(keys,'');row.update(panel=panel,label=label)
    return row

def prepare_mine(dest,release_map):
    rows=[]
    from obspy import read
    for panel,label,cls,sensor,stamp,guide_start,guide_pdf in MINE:
        entry=base_row(panel,label)
        local=one('sensor_'+sensor+'_'+stamp+'*.npy')
        seed=one(sensor+'.'+stamp[:8]+'.*.Z.miniseed')
        rec=np.load(str(local),allow_pickle=True).item()
        tr=read(str(seed))[0]
        assert tr.stats.channel=='DHZ' and tr.stats.sampling_rate==FS
        with tempfile.TemporaryDirectory() as folder:
            # The project entry point checks the filename suffix. Only the alias changes.
            alias=Path(folder)/seed.name.replace('.Z.miniseed','.DHZ.miniseed')
            alias.write_bytes(seed.read_bytes())
            filtered=preprocess(alias)
        utc=str(rec['event_start']).rstrip('Z')
        start=UTCDateTime(utc)-2
        offset=round(float(start-tr.stats.starttime)*FS)
        x=np.asarray(rec['signal'])
        cut=filtered.data[offset:offset+len(x)]
        if not np.array_equal(cut,x): raise ValueError('Filtered raw trace differs from '+local.name)
        if guide_pdf:
            pos=round((UTCDateTime(guide_start)-tr.stats.starttime)*FS)
            raw=tr.data[pos:pos+2501]
            entry['guide_correlation']=corr(guide_curve(SRC/guide_pdf,2501),raw)
            assert entry['guide_correlation']>.995
        rid=release_map.get(digest(local),'')
        if (panel=='d' and rid) or (panel!='d' and not rid): raise ValueError('Release map mismatch')
        entry.update(identifier=local.stem,trigger_utc=utc,record_start_utc=str(start),component='DHZ',
            native_unit='counts',source_status='guide_folder_processed' if not rid else 'final_release',
            guide_pdf=guide_pdf,guide_start_utc=guide_start,release_relative_path=rid,original_source=local.name,
            original_sha256=digest(local),processing_input=seed.name,processing_input_sha256=digest(seed),
            sample_offset=offset,raw_filter_max_abs_error=float(np.max(abs(cut-x))))
        rows.append(save_record(dest,entry,x,rec['time_axis']-rec['time_axis'][0]))
    return rows

def prepare_slides(dest):
    from scipy.io import loadmat
    from slide_match import template, measure
    info=json.loads((SRC/'mat_provenance.json').read_text(encoding='utf8'))
    mat=SRC/'quake_10s.mat'
    if digest(mat)!=info['mat_sha256']: raise ValueError('MAT checksum mismatch')
    bank=loadmat(str(mat))['quake_10s']
    assert bank.shape==(239,2500,6) and np.isfinite(bank).all()
    mean=bank[:,:,1].mean(axis=0)
    original_corr=corr(mean,guide_curve(SRC/'滑坡示例图.pdf',2500))
    if original_corr<.99999: raise ValueError('Local slide PDF is not the channel-2 mean')
    def filt(x): return bandpass(x,1.,20.,FS,corners=4,zerophase=True)
    guide=template(SRC/'判别依据.png')
    ranks=[]
    for i,sample in enumerate(bank):
        for ch,x in enumerate(sample.T):
            y=filt(x)
            score,env=measure(y,guide)
            ranks.append((score,i,ch,env))
    ranks.sort(key=lambda r:(-r[0],r[1],r[2]))
    with (dest/'slide_ranking.csv').open('w',encoding='utf-8-sig',newline='') as f:
        wr=csv.writer(f);wr.writerow(['rank','matlab_record','matlab_channel','match_score','envelope_cosine'])
        for k,(s,i,ch,env) in enumerate(ranks,1):wr.writerow([k,i+1,ch+1,s,env])
    outputs=[]
    for key,winner,pdf in [('c',ranks[0],'判别依据.png')]:
        score,i,ch=winner[:3]
        row=base_row(key,'Small landslide')
        row.update(identifier='quake_10s[%d,:,%d] (MATLAB)'%(i+1,ch+1),component='MAT channel %d'%(ch+1),
            native_unit='source unit unspecified',source_status='external_single_record',guide_pdf=pdf,
            original_source=mat.name,original_sha256=digest(mat),processing_input=mat.name,processing_input_sha256=digest(mat),
            matlab_record=i+1,matlab_channel=ch+1)
        row['guide_match_score']=float(score)
        row['guide_envelope_cosine']=float(winner[3])
        outputs.append(save_record(dest,row,filt(bank[i,:,ch]),np.arange(bank.shape[1])/FS))
    return outputs

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,default=HERE/'records')
    dest=p.parse_args().out
    dest.mkdir(parents=True,exist_ok=True)
    with (SRC/'source_release_map.csv').open(encoding='utf-8-sig') as f:
        released={r['sha256']:r['release_path'] for r in csv.DictReader(f)}
    rows=prepare_mine(dest,released)+prepare_slides(dest)
    rows.sort(key=lambda r:'abcde'.find(r['panel']) if len(r['panel'])==1 else 5)
    with (dest/'sources.csv').open('w',newline='',encoding='utf-8-sig') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print('Rebuilt the four mine records and the external landslide window.')

if __name__=='__main__': main()
