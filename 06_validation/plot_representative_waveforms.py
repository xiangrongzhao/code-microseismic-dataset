"""Select and plot real, labelled waveform examples without changing source data.

Scan uses waveform morphology only to rank candidates within existing class folders.
Final choices are supplied as a JSON manifest with source paths and SHA-256 checksums.
NPY dictionaries must come from a trusted source because loading uses pickle.
"""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
import argparse, csv, hashlib, json, sys
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, welch, spectrogram
import scipy
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CLASSES=['microseismic','blasting','small_landslide','mechanical_mining','transport_vehicles']
NAMES=['Microseismic','Blasting','Small landslide','Mechanical mining','Transport vehicles']
COLORS=['#307F83','#2862A3','#E59445','#744399','#777777']
REQUIRED={'sensor_id','event_start','event_end','time_axis','signal','sampling_rate'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.titlesize':11,'axes.labelsize':9,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.spines.top':False,'axes.spines.right':False,
 'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})

def load(path,expected_hash=None):
    payload=Path(path).read_bytes();digest=hashlib.sha256(payload).hexdigest()
    if expected_hash and digest!=expected_hash:raise ValueError('Source checksum mismatch: '+str(path))
    item=np.load(BytesIO(payload),allow_pickle=True).item()
    if not isinstance(item,dict) or set(item)!=REQUIRED:raise ValueError('Unexpected dictionary fields')
    x=np.asarray(item['signal'],dtype=float);t=np.asarray(item['time_axis'],dtype=float);fs=float(item['sampling_rate'])
    if x.ndim!=1 or t.shape!=x.shape or len(x)<256 or not np.isfinite(x).all() or not np.isfinite(t).all():raise ValueError('Invalid waveform')
    if fs<=0 or not np.isfinite(fs) or not np.allclose(np.diff(t),1/fs,rtol=1e-6,atol=1e-10):raise ValueError('Invalid sampling/time axis')
    if np.max(np.abs(x))<=0:raise ValueError('Zero waveform')
    return item,x,t-t[0],fs,digest

def measure(path,root):
    try:
        item,x,t,fs,digest=load(path)
        # Mean removal here affects candidate ranking only; output waveforms retain x.
        z=x-np.mean(x);z=z/(np.max(np.abs(z))+1e-30)
        env=np.sqrt(gaussian_filter1d(z*z,max(1,.08*fs)));env/=env.max()+1e-30
        idx=np.arange(0,len(env),max(1,round(fs/25)));e=env[idx];tt=t[idx]
        peak=int(np.argmax(e));peak_t=float(tt[peak]);energy=np.cumsum(z*z);energy/=energy[-1]
        t05,t50,t95=[float(t[np.searchsorted(energy,q)]) for q in [.05,.5,.95]]
        peaks,props=find_peaks(e,distance=max(1,round(.6*25)),prominence=.15,height=.28)
        intervals=np.diff(tt[peaks]);reg=float(1/(1+np.std(intervals)/np.mean(intervals))) if len(intervals)>=2 else 0.
        troughs=[float(np.min(e[a:b+1])) for a,b in zip(peaks[:-1],peaks[1:])]
        valley=float(np.mean(troughs)) if troughs else 1.
        before=np.flatnonzero(e[:peak+1]<.2)
        rise=peak_t-float(tt[before[-1]]) if len(before) else peak_t
        freq,p=welch(z,fs=fs,window='hann',nperseg=min(512,len(z)),noverlap=min(384,len(z)//2),nfft=1024,detrend='constant',scaling='density')
        mask=(freq>=.5)&(freq<=40);pp=p[mask];ff=freq[mask];pp/=pp.sum()+1e-30
        fpeak=float(ff[np.argmax(pp)]);centroid=float(np.sum(ff*pp));narrow=float(pp[np.abs(ff-fpeak)<=1.5].sum())
        pre=float(np.sqrt(np.mean(z[t<1]**2)));post=float(np.sqrt(np.mean(z[t>t[-1]-1]**2)))
        peak_rms=float(np.sqrt(np.max(gaussian_filter1d(z*z,max(1,.08*fs)))))
        occupancy=float(np.mean(e>.25));cv=float(np.std(e)/(np.mean(e)+1e-30));active=t95-t05
        contrast=np.clip(np.log10((peak_rms+1e-10)/(pre+1e-10))/2,0,1)
        tailquiet=np.clip(1-post/(peak_rms+1e-10),0,1)
        central=np.exp(-((peak_t-3)/2.5)**2)
        # Fixed transparent, descriptive scores; they are not class probabilities.
        scores={
          'microseismic':2*contrast+tailquiet+central+np.exp(-((active-3)/2.5)**2)+.5*np.clip(rise/.7,0,1)-.10*max(0,len(peaks)-4),
          'blasting':2*contrast+1.5*tailquiet+np.exp(-rise/.6)+np.exp(-((peak_t-2)/1.8)**2)+np.exp(-((active-2)/2)**2)-.15*max(0,len(peaks)-3),
          'small_landslide':1.6*contrast+.8*tailquiet+np.clip(rise/1.4,0,1)+np.exp(-((active-5)/2.5)**2)+np.exp(-((peak_t-4)/2.2)**2),
          'mechanical_mining':2.5*reg+2*(1-valley)+np.exp(-((len(peaks)-4)/2)**2)+.5*occupancy-1.5*(len(peaks)<3),
          'transport_vehicles':2*occupancy+2/(1+3*cv)+narrow+np.clip(active/8,0,1)-.2*np.clip(contrast,0,1),
        }
        c=path.parent.name
        # Favor a complete available 10-second window, without padding or resampling.
        quality=float(min(1,(t[-1]+1/fs)/9.8))
        return {'class':c,'relative_path':path.relative_to(root).as_posix(),'sha256':digest,'sensor_id':str(item['sensor_id']),
          'event_start':str(item['event_start']),'event_end':str(item['event_end']),'samples':len(x),'sampling_rate':fs,'duration_s':float(t[-1]),
          'score':float(scores[c]*quality),'peak_time_s':peak_t,'rise_s':float(rise),'energy_90_duration_s':active,'energy_05_s':t05,
          'energy_95_s':t95,'envelope_peaks':len(peaks),'peak_spacing_regularity':reg,'mean_valley':valley,'occupancy':occupancy,
          'envelope_cv':cv,'peak_frequency_hz':fpeak,'spectral_centroid_hz':centroid,'narrow_fraction':narrow,'pre_rms':pre,'post_rms':post}
    except Exception as exc:return {'relative_path':path.relative_to(root).as_posix(),'error':str(exc)}

def scan(args):
    root=args.data_root.resolve();files=[p for c in CLASSES for p in sorted((root/c).glob('*.npy'))]
    records=[];errors=[]
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for i,result in enumerate(pool.map(lambda p:measure(p,root),files),1):
            (errors if 'error' in result else records).append(result)
            if i%2000==0:print(f'Scanned {i}/{len(files)}',flush=True)
    with (args.out/'candidate_metrics.csv').open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(records[0]));writer.writeheader();writer.writerows(records)
    counts={c:sum(r['class']==c for r in records) for c in CLASSES}
    (args.out/'scan_summary.json').write_text(json.dumps({'counts':counts,'total':len(records),'errors':errors,'source_data_modified':False},indent=2),encoding='utf-8')
    if errors:raise RuntimeError(f'{len(errors)} files could not be scanned; see scan_summary.json')
    candidates(args,records)

def candidates(args,records=None):
    if records is None:
        with (args.out/'candidate_metrics.csv').open(encoding='utf-8-sig') as f:records=list(csv.DictReader(f))
    shortlist={}
    for c,name,color in zip(CLASSES,NAMES,COLORS):
        ranked=sorted([r for r in records if r['class']==c],key=lambda r:float(r['score']),reverse=True)
        if c=='transport_vehicles' and args.min_vehicle_peak_hz>0:
            ranked=[r for r in ranked if float(r['peak_frequency_hz'])>=args.min_vehicle_peak_hz]
        # Avoid spending the whole contact sheet on multiple stations of one trigger cluster.
        chosen=[];seen=set()
        for r in ranked:
            timebin=str(r['event_start'])[:16]
            if timebin in seen:continue
            chosen.append(r);seen.add(timebin)
            if len(chosen)>=args.candidates:break
        shortlist[c]=chosen
        fig,axs=plt.subplots(len(chosen),2,figsize=(13,2.05*len(chosen)),gridspec_kw={'width_ratios':[2.2,1]})
        for i,r in enumerate(chosen):
            _,x,t,fs,_=load(args.data_root/r['relative_path'],r['sha256']);xn=x/np.max(np.abs(x))
            env=np.sqrt(gaussian_filter1d(xn*xn,max(1,.08*fs)))
            axs[i,0].plot(t,xn,color=color,lw=.55);axs[i,0].plot(t,env,color='black',lw=.55,alpha=.5)
            axs[i,0].set_xlim(0,t[-1]);axs[i,0].set_ylim(-1.08,1.08)
            axs[i,0].set_title(f"#{i+1} | {r['sensor_id']} | {r['event_start']} | score={float(r['score']):.2f}",fontsize=9,loc='left')
            freq,st,psd=compute_tf(xn,fs)
            axs[i,1].pcolormesh(st,freq,10*np.log10(np.maximum(psd,1e-12)),shading='auto',cmap='magma',vmin=-65,vmax=-5,rasterized=True)
            axs[i,1].set_ylim(0,30);axs[i,1].set_ylabel('Hz');axs[i,1].set_xlim(0,t[-1])
            axs[i,1].set_title(f"rise={float(r['rise_s']):.2f}s; duration90={float(r['energy_90_duration_s']):.2f}s; peaks={int(r['envelope_peaks'])}",fontsize=8)
        fig.suptitle(name+' - real waveform candidates (normalized)',fontsize=14,y=1)
        fig.tight_layout();fig.savefig(args.out/f'candidates_{c}.png',dpi=140,bbox_inches='tight');plt.close(fig)
    (args.out/'shortlist.json').write_text(json.dumps(shortlist,indent=2),encoding='utf-8')
    print('Candidate plots and shortlist saved.',flush=True)

def compute_tf(x,fs):
    return spectrogram(x,fs=fs,window='hann',nperseg=128,noverlap=112,nfft=512,
       detrend='constant',scaling='density',mode='psd')

def save_vector(fig,path,formats=('png','svg'),dpi=400,transparent=False):
    for fmt in formats:fig.savefig(path.with_suffix('.'+fmt),dpi=dpi,bbox_inches='tight',pad_inches=.035,transparent=transparent)

def plot(args):
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    if set(selection)!=set(CLASSES):raise ValueError('Selection must contain exactly the five class keys')
    panels=args.out/'panels';panels.mkdir(exist_ok=True)
    fig,axs=plt.subplots(3,5,figsize=(15.8,8.3),gridspec_kw={'height_ratios':[1,1.05,.92]})
    fig.subplots_adjust(left=.07,right=.92,bottom=.13,top=.9,wspace=.32,hspace=.48)
    export=[];spectra={};images=[]
    for j,(c,name,color) in enumerate(zip(CLASSES,NAMES,COLORS)):
        entry=selection[c];path=args.data_root/entry['relative_path']
        item,x,t,fs,digest=load(path,entry['sha256']);scale=float(np.max(np.abs(x)));xn=x/scale
        freq,st,p=compute_tf(xn,fs);db=10*np.log10(np.maximum(p,1e-12))
        wf,wp=welch(xn,fs=fs,window='hann',nperseg=min(512,len(xn)),noverlap=min(384,len(xn)//2),nfft=1024,detrend='constant',scaling='density')
        spectra.update({c+'_tf_frequency_hz':freq,c+'_tf_time_s':st,c+'_tf_psd_per_hz':p,c+'_welch_frequency_hz':wf,c+'_welch_psd_per_hz':wp})
        axs[0,j].plot(t,x,color=color,lw=.65);axs[0,j].set_xlim(0,t[-1]);axs[0,j].set_xlabel('Time (s)')
        axs[0,j].set_title(f'({chr(97+j)}) {name}',fontsize=10.7,fontweight='bold',pad=19)
        axs[0,j].ticklabel_format(axis='y',style='sci',scilimits=(-2,3),useOffset=False)
        im=axs[1,j].pcolormesh(st,freq,db,shading='auto',cmap='magma',vmin=-65,vmax=-5,rasterized=True);images.append(im)
        axs[1,j].set_ylim(0,30);axs[1,j].set_xlim(0,t[-1]);axs[1,j].set_xlabel('Time (s)');axs[1,j].set_yticks([0,10,20,30])
        axs[2,j].plot(wf,10*np.log10(np.maximum(wp,1e-12)),color=color,lw=1);axs[2,j].set_xlim(0,30);axs[2,j].set_ylim(-80,0);axs[2,j].set_xlabel('Frequency (Hz)');axs[2,j].grid(alpha=.15)
        if j==0:
            axs[0,j].set_ylabel('Amplitude (stored units)');axs[1,j].set_ylabel('Frequency (Hz)');axs[2,j].set_ylabel('Normalized PSD (dB/Hz)')
        for kind in ['waveform','spectrogram','spectrum']:
            f,a=plt.subplots(figsize=(3.45,2.15))
            if kind=='waveform':
                a.plot(t,x,color=color,lw=.75);a.set_xlim(0,t[-1]);a.set_xlabel('Time (s)');a.set_ylabel('Amplitude (stored units)');a.ticklabel_format(axis='y',style='sci',scilimits=(-2,3),useOffset=False)
            elif kind=='spectrogram':
                m=a.pcolormesh(st,freq,db,shading='auto',cmap='magma',vmin=-65,vmax=-5,rasterized=True);a.set_ylim(0,30);a.set_xlim(0,t[-1]);a.set_xlabel('Time (s)');a.set_ylabel('Frequency (Hz)')
                cb=f.colorbar(m,ax=a,pad=.02);cb.set_label('Normalized PSD (dB/Hz)',fontsize=7);cb.ax.tick_params(labelsize=7)
            else:
                a.plot(wf,10*np.log10(np.maximum(wp,1e-12)),color=color);a.set_xlim(0,fs/2);a.set_ylim(-120,0);a.set_xlabel('Frequency (Hz)');a.set_ylabel('Normalized PSD (dB/Hz)');a.grid(alpha=.15)
            a.set_title(name);f.tight_layout();save_vector(f,panels/(c+'_'+kind));plt.close(f)
        # Borderless tiles for the compact workflow diagram. Same full record, scalar normalization only.
        f,a=plt.subplots(figsize=(2.8,.78));a.plot(t,xn,color=color,lw=.65);a.set_xlim(0,t[-1]);a.set_ylim(-1.08,1.08);a.axis('off');f.subplots_adjust(0,0,1,1);save_vector(f,panels/(c+'_waveform_tile'),transparent=True);plt.close(f)
        f,a=plt.subplots(figsize=(2.8,1.02));a.pcolormesh(st,freq,db,shading='auto',cmap='magma',vmin=-65,vmax=-5,rasterized=True);a.set_ylim(0,30);a.set_xlim(0,t[-1]);a.set_facecolor('white');a.axis('off');f.subplots_adjust(0,0,1,1);save_vector(f,panels/(c+'_spectrogram_tile'));plt.close(f)
        record={**entry,'class':c,'sensor_id':str(item['sensor_id']),'event_start':str(item['event_start']),'event_end':str(item['event_end']),
          'samples':len(x),'sampling_rate_hz':fs,'display_duration_s':float(t[-1]),'peak_normalization_factor':scale,
          'component':'vertical (Z), according to released event-file documentation','waveform_processing':'none; stored signal samples',
          'spectral_processing':'divide by max(abs(signal)); Hann window; constant detrend per segment; PSD density',
          'stft_nperseg':128,'stft_noverlap':112,'stft_nfft':512,'psd_display_min_db':-65,'psd_display_max_db':-5}
        export.append(record)
        assert hashlib.sha256(path.read_bytes()).hexdigest()==digest
    cax=fig.add_axes([.94,.405,.01,.205]);cb=fig.colorbar(images[0],cax=cax,ticks=[-65,-45,-25,-5]);cb.set_label('PSD of peak-normalized waveform\n(dB re 1 Hz$^{-1}$)',fontsize=9)
    fig.text(.5,.026,'Waveforms: stored vertical-component samples. Spectra: peak-normalized records; common time-frequency color scale.',ha='center',fontsize=9)
    save_vector(fig,args.out/'five_class_waveforms_and_spectra',formats=('png','svg','pdf'),dpi=350);plt.close(fig)
    # One compact replacement strip with a caption but without extra processing.
    f,aa=plt.subplots(2,5,figsize=(10.4,2.6),gridspec_kw={'height_ratios':[.7,1]})
    for j,(c,name,color) in enumerate(zip(CLASSES,NAMES,COLORS)):
        _,x,t,fs,_=load(args.data_root/selection[c]['relative_path'],selection[c]['sha256']);xn=x/np.max(np.abs(x));fr,st,p=compute_tf(xn,fs)
        aa[0,j].plot(t,xn,color=color,lw=.7);aa[0,j].set_ylim(-1.05,1.05);aa[0,j].set_xlim(0,t[-1]);aa[0,j].axis('off');aa[0,j].set_title(name.replace(' ','\n') if j>=2 else name,fontsize=10)
        aa[1,j].pcolormesh(st,fr,10*np.log10(np.maximum(p,1e-12)),shading='auto',cmap='magma',vmin=-65,vmax=-5,rasterized=True);aa[1,j].set_xlim(0,t[-1]);aa[1,j].set_ylim(0,30);aa[1,j].set_facecolor('white');aa[1,j].axis('off')
    f.subplots_adjust(left=.015,right=.985,top=.74,bottom=.15,wspace=.12,hspace=.10)
    f.text(.5,.03,'Each record: approximately 10 s | Frequency: 0-30 Hz | Shared normalized PSD scale: -65 to -5 dB/Hz',ha='center',fontsize=8)
    save_vector(f,args.out/'five_class_workflow_strip',dpi=500);plt.close(f)
    np.savez_compressed(args.out/'computed_spectra.npz',**spectra)
    (args.out/'selected_sources.json').write_text(json.dumps(export,indent=2,ensure_ascii=False),encoding='utf-8')
    with (args.out/'selected_sources.csv').open('w',encoding='utf-8-sig',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(export[0]));writer.writeheader();writer.writerows(export)
    (args.out/'software_versions.json').write_text(json.dumps({'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'matplotlib':matplotlib.__version__},indent=2),encoding='utf-8')
    print('Plotted all five real records. Source bytes unchanged.',flush=True)

def plot_text_free(args):
    """One text-free waveform/spectrogram pair per class, using the fixed records."""
    selection=json.loads(args.selection.read_text(encoding='utf-8'))
    if set(selection)!=set(CLASSES):raise ValueError('Selection must contain exactly the five class keys')
    for c,color in zip(CLASSES,COLORS):
        entry=selection[c]
        _,x,t,fs,digest=load(args.data_root/entry['relative_path'],entry['sha256'])
        xn=x/np.max(np.abs(x));freq,st,p=compute_tf(xn,fs)
        fig=plt.figure(figsize=(3.6,2.35))
        wave=fig.add_axes([.025,.61,.95,.365])
        wave.plot(t,xn,color=color,lw=.75)
        wave.set_xlim(0,t[-1]);wave.set_ylim(-1.08,1.08);wave.axis('off')
        tf=fig.add_axes([.025,.025,.95,.535])
        tf.pcolormesh(st,freq,10*np.log10(np.maximum(p,1e-12)),shading='auto',
                      cmap='magma',vmin=-65,vmax=-5,rasterized=True)
        tf.set_xlim(0,t[-1]);tf.set_ylim(0,30);tf.set_facecolor('white');tf.axis('off')
        # Fixed canvas keeps all five output sizes identical. No text artists are added.
        for fmt in ['png','svg']:
            fig.savefig(args.out/(c+'.'+fmt),dpi=500,facecolor='white',bbox_inches=None,pad_inches=0)
        plt.close(fig)
        assert hashlib.sha256((args.data_root/entry['relative_path']).read_bytes()).hexdigest()==digest
    print('Exported five text-free waveform/spectrogram pairs (PNG and SVG).',flush=True)

def main():
    a=argparse.ArgumentParser(description=__doc__)
    a.add_argument('--data-root',type=Path,required=True);a.add_argument('--out',type=Path,required=True)
    a.add_argument('--stage',choices=['scan','candidates','plot','text-free'],required=True);a.add_argument('--selection',type=Path)
    a.add_argument('--trust-source',action='store_true');a.add_argument('--workers',type=int,default=4);a.add_argument('--candidates',type=int,default=10)
    a.add_argument('--min-vehicle-peak-hz',type=float,default=0,help='Optional display-oriented candidate filter; does not modify records or labels')
    args=a.parse_args()
    if not args.trust_source:a.error('--trust-source is required for pickle-bearing NPY dictionaries')
    if args.out.resolve()==args.data_root.resolve() or args.data_root.resolve() in args.out.resolve().parents:a.error('Outputs must be outside the waveform dataset')
    args.out.mkdir(parents=True,exist_ok=True)
    if args.stage=='scan':scan(args)
    elif args.stage=='candidates':candidates(args)
    else:
        if args.selection is None:a.error('--selection is required for plotting')
        if args.stage=='text-free':plot_text_free(args)
        else:plot(args)
if __name__=='__main__':main()
