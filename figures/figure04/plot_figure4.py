"""Draw Figure 4 from verified filtered waveforms. Requires NumPy >=1.20 and Matplotlib."""
import argparse
import csv
import hashlib
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import numpy as np

HERE=Path(__file__).resolve().parent
FS=250.
COLORS=('#187fb5','#cc5048')

def load_records(folder):
    out={}
    with (folder/'sources.csv').open(encoding='utf-8-sig',newline='') as f:
        for row in csv.DictReader(f):
            with np.load(str(folder/(row['panel']+'.npz')),allow_pickle=False) as rec:
                x=rec['signal'];t=rec['time_axis']
                assert float(rec['sampling_rate'])==FS and len(x)==len(t)==int(row['n'])
                assert str(rec['source_id'])==row['identifier']
            if hashlib.sha256(x.astype('<f8').tobytes()).hexdigest()!=row['signal_sha256']:
                raise ValueError('Unexpected samples in panel '+row['panel'])
            out[row['panel']]=(t,x,row)
    return out

def spectrogram(x):
    n,hop,nfft=128,16,512
    w=.5-.5*np.cos(2*np.pi*np.arange(n)/n)
    centers=np.arange(0,len(x),hop)
    psd=np.empty((len(centers),nfft//2+1))
    for i,center in enumerate(centers):
        left=center-n//2
        lo,hi=max(left,0),min(left+n,len(x))
        samples=x[lo:hi]
        # At the boundaries use only observed samples and their Hann weights.
        weights=w[lo-left:hi-left]
        spectrum=np.fft.rfft((samples-samples.mean())*weights,n=nfft)
        psd[i]=np.abs(spectrum)**2/(FS*np.dot(weights,weights))
    psd[:,1:-1]*=2.
    f=np.fft.rfftfreq(nfft,1/FS);t=centers/FS
    peak=psd[:,(f>=1)&(f<=20)].max()
    db=10*np.log10(np.maximum(psd.T/peak,1.e-12))
    return f,t,db

def characteristic(x):
    # Calculating on relative amplitudes keeps divisions stable for small MAT values.
    x=x/np.max(abs(x))
    cs=np.r_[0.,np.cumsum(x*x)];end=np.arange(750,len(x)+1)
    short=(cs[end]-cs[end-125])/125;long=(cs[end]-cs[end-750])/750
    v=np.divide(short,long,out=np.zeros_like(short),where=long>0)
    return end-1,v/v.max() if np.any(v) else v

def tick_text(v,pos):
    if abs(v)>=1 and float(v).is_integer():
        return str(int(v)).replace('-','−')
    return ('%.3g'%v).replace('e-0','e−').replace('e+0','e+').replace('-','−')

def nice_limit(x):
    val=np.max(np.abs(x));power=10**np.floor(np.log10(val))
    return next(m*power for m in (1,1.25,1.5,2,2.5,3,4,5,6,8,10) if m*power>=val)

def figure(records, panels, native=False):
    count=len(panels);single=count==1
    width,height=(2.62,4.45) if single else (12.5,4.5)
    fig=plt.figure(figsize=(width,height),facecolor='white')
    w=.61/count
    rows=((.755,.175),(.455,.176),(.155,.177))
    positions=[]
    for col,key in enumerate(panels):
        t,x,meta=records[key]
        origin=(col+.213)/count
        axes=[fig.add_axes((origin,y,w,h)) for y,h in rows]
        positions.append([a.get_position().bounds for a in axes])
        wave,spec,sta=axes
        for ax,title in zip(axes,('Waveform','Time–frequency','Characteristic function')):
            ax.set_title(title,fontsize=9.0,pad=4.0)
            ax.set_xlim(0,10);ax.set_xticks([0,2,4,6,8,10])
            ax.tick_params(labelsize=7.8,width=.5,length=2,pad=1.4)
            for edge in ax.spines.values():
                edge.set_color('#676767');edge.set_linewidth(.48)
        for ax in (wave,sta):
            ax.set_axisbelow(True);ax.grid(color='#dedede',linewidth=.35)
        peak=np.max(np.abs(x))
        wave.plot(t,x if native else x/peak,color=COLORS[0],lw=.52)
        lim=nice_limit(x) if native else 1.
        wave.set_ylim(-1.075*lim,1.075*lim);wave.set_yticks([-lim,0,lim])
        wave.yaxis.set_major_formatter(FuncFormatter(tick_text))
        if native:
            unit='counts' if meta['native_unit']=='counts' else 'source units'
            wave.set_title('Waveform ('+unit+')',fontsize=8.8,pad=4)
            wave.set_ylabel('Amp.',fontsize=8.1,labelpad=2.4)
        else:
            wave.set_ylabel('Norm. amp.',fontsize=8.1,labelpad=2.4)
        f,ts,db=spectrogram(x);df=f[1]-f[0];dt=ts[1]-ts[0]
        im=spec.imshow(db,origin='lower',aspect='auto',interpolation='nearest',cmap='jet',vmin=-40,vmax=0,
            extent=[ts[0]-dt/2,ts[-1]+dt/2,f[0]-df/2,f[-1]+df/2],rasterized=True)
        spec.set_ylim(1,20);spec.set_yticks([1,10,20])
        spec.set_ylabel('Freq. (Hz)',fontsize=8.1,labelpad=2.4)
        cax=fig.add_axes(((col+.85)/count,rows[1][0],.023/count,rows[1][1]))
        cb=fig.colorbar(im,cax=cax,ticks=[-40,-20,0])
        cb.ax.set_title('dB',fontsize=7.6,pad=3)
        cb.ax.tick_params(labelsize=7.0,length=1.6,width=.42,pad=1.4)
        cb.outline.set_linewidth(.42)
        ix,z=characteristic(x)
        sta.plot(t[ix],z,color=COLORS[1],lw=.68)
        sta.set_ylim(-.025,1.04);sta.set_yticks([0,.5,1]);sta.set_yticklabels(['0','0.5','1'])
        sta.set_ylabel('Norm. STA/LTA',fontsize=8.1,labelpad=2.4)
        sta.set_xlabel('Time (s)',fontsize=8.1,labelpad=1.2)
        caption='('+key[0]+') '+meta['label']
        fig.text(origin+w/2,.037,caption,ha='center',va='center',fontsize=10.6)
    # Equal plot boxes and equal time-axis positions are enforced in both directions.
    box=np.array(positions)
    np.testing.assert_allclose(box[:,:,2],np.full((count,3),w))
    for col in range(count): np.testing.assert_allclose(box[col,:,0],np.repeat(box[col,0,0],3))
    for i in range(3):np.testing.assert_allclose(box[:,i,1],np.repeat(rows[i][0],count))
    return fig

def save(fig,folder,stem,formats):
    for fmt in formats:fig.savefig(str(folder/(stem+'.'+fmt)),dpi=450,facecolor='white')
    plt.close(fig)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--records',type=Path,default=HERE/'records')
    p.add_argument('--out',type=Path,default=HERE/'figures')
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True)
    plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'pdf.fonttype':42,'axes.unicode_minus':True})
    records=load_records(a.records);panels=list('abcde')
    for key in panels:
        panel = figure(records,[key])
        # The manuscript supplies the caption below each panel.
        for annotation in list(panel.texts):
            annotation.remove()
        panel.savefig(str(a.out/('fig4-'+key+'.pdf')),dpi=450,
                      bbox_inches=matplotlib.transforms.Bbox.from_bounds(.177,.39,2.37,3.90),
                      facecolor='white')
        plt.close(panel)
    print('Figure 4 written to '+str(a.out))

if __name__=='__main__':main()
