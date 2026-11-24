"""Match a measured, filtered trace to the red small-landslide panel supplied by the author."""
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import spectrogram

BINS=40
U=(np.arange(BINS)+.5)/BINS
F=np.arange(1.,20.1,.5)

def cosine(a,b):
    a,b=np.ravel(a),np.ravel(b)
    return np.clip(np.dot(a,b)/max(np.linalg.norm(a)*np.linalg.norm(b),1.e-25),0,1)

def bin_profile(x,rms=False):
    cuts=np.linspace(0,len(x),BINS+1).astype(int)
    value=np.add.reduceat(x*x if rms else x,cuts[:-1])/np.diff(cuts)
    return np.sqrt(np.maximum(0,value)) if rms else value

def red(rgb):
    return np.maximum(0,np.minimum(rgb[:,:,0]-rgb[:,:,1],rgb[:,:,0]-rgb[:,:,2])-4)

def template(path):
    rgba=Image.open(str(path)).convert('RGBA')
    if rgba.size!=(856,187):raise ValueError('Unexpected author-supplied guide dimensions')
    rgb=Image.new('RGB',rgba.size,'white');rgb.paste(rgba,mask=rgba.getchannel('A'))
    img=np.array(rgb,dtype=float)
    # Slide panel calibration on the 856 x 187 reference, with axes excluded.
    x0,x1,top,end=374,510,39,65
    zero,hi,fmax,cb,ct=106.5,79,10.,148.,120
    weights=red(img[top:end+1,x0:x1]);ys=np.arange(top,end+1)[:,None]
    den=weights.sum(axis=0)
    center=(ys*weights).sum(axis=0)/np.maximum(den,1)
    center=np.interp(np.arange(len(den)),np.flatnonzero(den),center[den>0])
    raw=(weights*(ys-gaussian_filter1d(center,3.6))**2).sum(axis=0)/np.maximum(den,1)
    env=bin_profile(np.sqrt(np.maximum(0,raw-min(np.quantile(raw,.10),.8))),True)
    weights=red(img[ct:int(cb)+2,x0:x1])
    h=((cb-np.arange(ct,int(cb)+2))[:,None]*weights).sum(axis=0)/np.maximum(weights.sum(axis=0),1)
    cf=bin_profile(np.maximum(0,h))
    patch=img[int(hi)+1:int(zero),x0:x1]
    contrast=np.maximum(0,np.linalg.norm(patch-np.median(patch,axis=1,keepdims=True),axis=2)-6)
    xx,yy=np.meshgrid(U*(x1-x0)-.5,zero-(zero-hi)*F/fmax-(hi+1))
    power=map_coordinates(contrast,[yy,xx],order=1,mode='constant')
    return env,cf,gaussian_filter1d(gaussian_filter1d(power,1.25,axis=0),.8,axis=1)

def measure(x,ref):
    if not np.isfinite(x).all() or np.ptp(x)==0: raise ValueError('Invalid MAT trace')
    env,cf,power=ref
    e=bin_profile(x,True)
    fine=cosine(e,env)
    coarse=cosine(gaussian_filter1d(e,2),gaussian_filter1d(env,2))
    contrast=abs(np.std(e)/e.mean()-np.std(env)/env.mean())
    shape=.65*fine+.22*coarse+.13*np.exp(-contrast)
    end=np.arange(750,len(x)+1);cs=np.r_[0.,np.cumsum(x*x)]
    sta=(cs[end]-cs[end-125])/125;lta=(cs[end]-cs[end-750])/750
    ratio=np.divide(sta,lta,out=np.zeros_like(sta),where=lta>0)
    v=np.interp(U*len(x),end-1,ratio)[U>.30]
    c=cf[U>.30]
    characteristic=cosine(np.maximum(0,v-np.quantile(v,.1)),np.maximum(0,c-np.quantile(c,.1)))
    f,t,p=spectrogram(x,fs=250.,window='hann',nperseg=250,noverlap=225,nfft=500)
    tgrid=np.clip((U*len(x)/250.-t[0])/(t[1]-t[0]),0,len(t)-1)
    left=tgrid.astype(int);px=p[(f>=1)&(f<=20)]
    interp=px[:,left]*(1-(tgrid-left))+px[:,np.minimum(left+1,len(t)-1)]*(tgrid-left)
    # Match only the reliable overlapping portion of the digitized frequency scale.
    spectral=cosine(np.sqrt(interp[F<=9]),power[F<=9])
    center=F[np.argmax(power.sum(axis=1))]
    packet_band=np.abs(F-center)<=1.
    target_packet=power[packet_band].sum(axis=0)
    measured_packet=interp[packet_band].sum(axis=0)
    packet_timing=cosine(measured_packet,target_packet)
    # Account for 1–20 Hz energy outside the reference's low-frequency packet.
    # A well-timed high-frequency burst alone is a poor match to this slide example.
    local_spectrum=interp @ (target_packet**2/np.sum(target_packet**2))
    packet_fraction=local_spectrum[packet_band].sum()/local_spectrum.sum()
    score=.28*shape+.12*characteristic+.18*spectral+.22*packet_timing+.20*packet_fraction
    return float(score),float(fine)
