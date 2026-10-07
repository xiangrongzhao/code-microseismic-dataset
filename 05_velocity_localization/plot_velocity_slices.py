"""Plot Vs sections from XJD2_updated.txt, with depth below each model column top."""
from pathlib import Path
import argparse, hashlib, json
import numpy as np
from scipy.interpolate import RegularGridInterpolator
from pyproj import Proj, Geod
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from mpl_toolkits.mplot3d import proj3d

PROJ = '+proj=tmerc +lat_0=0 +lon_0=96 +k=1 +x_0=500000 +y_0=0 +a=6378245.0 +b=6356863.0187730473 +units=m +no_defs'
BG = '#52576e'

def cluster(v, tolerance=1e-5):
    order=np.argsort(v); g=np.cumsum(np.r_[True,np.diff(v[order])>tolerance])-1
    ids=np.empty(len(v),int); ids[order]=g
    centres=np.array([np.median(v[ids==i]) for i in range(g.max()+1)])
    return centres,ids

def load_model(path):
    a=np.loadtxt(path)
    lon,lat=Proj(PROJ)(a[:,0]-32000000,a[:,1],inverse=True)
    lon_axis,ix=cluster(lon); lat_axis,iy=cluster(lat)
    top=np.full((len(lon_axis),len(lat_axis)),-np.inf)
    np.maximum.at(top,(ix,iy),a[:,2])
    depths=(top[ix,iy]-a[:,2])/1000
    depth_axis=np.unique(np.round(depths,6)); iz=np.searchsorted(depth_axis,np.round(depths,6))
    cube=np.full((*top.shape,len(depth_axis)),np.nan)
    assert len(np.unique(np.column_stack([ix,iy,iz]),axis=0))==len(a)
    cube[ix,iy,iz]=a[:,3]
    assert cube.shape==(26,26,10) and np.isfinite(cube).all()
    assert np.allclose(depth_axis,np.arange(10)*.1)
    assert cube.min()==.1 and cube.max()==1.0
    geod=Geod(ellps='WGS84')
    ew=geod.inv(lon.min(),lat.mean(),lon.max(),lat.mean())[2]/1000
    ns=geod.inv(lon.mean(),lat.min(),lon.mean(),lat.max())[2]/1000
    east=(lon_axis-lon_axis[0])/(lon_axis[-1]-lon_axis[0])*ew
    north=(lat_axis-lat_axis[0])/(lat_axis[-1]-lat_axis[0])*ns
    metadata={'source':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest(),
        'point_count':len(a),'grid_shape':list(cube.shape),'longitude_bounds':[float(lon.min()),float(lon.max())],
        'latitude_bounds':[float(lat.min()),float(lat.max())],'east_west_span_km':ew,'north_south_span_km':ns,
        'depth_km':depth_axis.tolist(),'depth_reference':'Depth below the upper surface of each model column, derived from column maximum Z minus Z.',
        'Vs_km_s':[float(cube.min()),float(cube.max())],'Vp_km_s_if_sqrt3':[float(cube.min()*np.sqrt(3)),float(cube.max()*np.sqrt(3))],
        'geographic_transform':PROJ,'geographic_note':'Historical inverse projection; an independent datum transformation is not established.',
        'interpolation':'Trilinear within the supplied model. No padding or extrapolated velocities.',
        'color_map':'jet_r, red at Vs=0.1 and blue at Vs=1.0',
        'plot_box_aspect':'Equal physical scale for east, north and depth.'}
    return east,north,depth_axis,cube,metadata

def dms_lon(v):
    sec=(v-94)*3600;minute=int(sec//60);return f'{minute:02d}′{sec-minute*60:04.1f}″'

def annotate_axes(fig,ax,east,north,depth,meta,azim):
    """Place labels in screen space to separate labels at shared 3-D corners."""
    E,N,D=east[-1],north[-1],depth[-1]
    if azim < 0:
        edges=[([0,0,D],[E,0,D]),([E,0,D],[E,N,D]),([E,N,0],[E,N,D])]
    else:
        edges=[([0,N,D],[E,N,D]),([E,0,D],[E,N,D]),([0,N,0],[0,N,D])]
    ax.set_axis_off()
    for start,end in edges:
        ax.plot(*np.asarray([start,end]).T,color='white',lw=.75)
    fig.canvas.draw()
    def project(point):
        q=proj3d.proj_transform(*point,ax.get_proj())
        return ax.transData.transform(q[:2])
    center=project([E/2,N/2,D/2])
    labels=[('Longitude (94° E)',[dms_lon(v) for v in np.linspace(*meta['longitude_bounds'],3)]),
            ('Latitude (43°55′ N)',[f'{(v-43-55/60)*3600:04.1f}″' for v in meta['latitude_bounds']]),
            ('Depth (km)',['0.0','0.3','0.6','0.9'])]
    for axis_index,((start,end),(title,ticks)) in enumerate(zip(edges,labels)):
        start,end=np.array(start),np.array(end)
        p0,p1=project(start),project(end);direction=(p1-p0)/np.linalg.norm(p1-p0)
        normal=np.array([-direction[1],direction[0]])
        if np.dot((p0+p1)/2-center,normal)<0:normal=-normal
        for j,t in enumerate(np.linspace(0,1,len(ticks))):
            base=project(start+t*(end-start))
            offset=normal*(11 if axis_index!=2 else 8)*fig.dpi/72
            # Shift horizontal endpoint labels towards the centre of their own axis.
            along=(26 if j==0 else -26 if j==len(ticks)-1 else 0)*fig.dpi/72 if axis_index==0 else 0
            loc=fig.transFigure.inverted().transform(base+offset+along*direction)
            fig.text(*loc,ticks[j],ha='center',va='center',color='white',fontsize=11)
        midpoint=project((start+end)/2)+normal*(31 if axis_index!=2 else 27)*fig.dpi/72
        loc=fig.transFigure.inverted().transform(midpoint)
        angle=np.degrees(np.arctan2(direction[1],direction[0]))
        if angle>90:angle-=180
        if angle<-90:angle+=180
        fig.text(*loc,title,ha='center',va='center',rotation=angle,color='white',fontsize=12)

def plot(model,out):
    out.mkdir(parents=True,exist_ok=True)
    east,north,depth,cube,meta=load_model(model)
    interp=RegularGridInterpolator((east,north,depth),cube,bounds_error=True)
    norm=Normalize(.1,1);cmap=plt.get_cmap('jet_r')
    plt.rcParams.update({'font.family':'serif','font.serif':['Times New Roman','DejaVu Serif'],
                         'font.size':11,'savefig.facecolor':BG,'svg.fonttype':'none'})
    settings=[('longitude_sections',(4.6,5.0),25,-57),
              ('latitude_sections',(4.6,5.0),24,36),
              ('depth_slices',(4.6,5.0),23,-58)]
    cuts={}
    for name,size,elev,azim in settings:
        fig=plt.figure(figsize=size,facecolor=BG)
        ax=fig.add_axes([.05,.13,.82,.78],projection='3d',facecolor=BG)
        ax.set_proj_type('ortho');ax.view_init(elev=elev,azim=azim)
        ax.set_xlim(0,east[-1]);ax.set_ylim(0,north[-1]);ax.set_zlim(depth[-1],0)
        ax.set_box_aspect((east[-1],north[-1],depth[-1]),zoom=.88)
        x=np.linspace(0,east[-1],96);y=np.linspace(0,north[-1],64);d=np.linspace(0,depth[-1],100)
        if name=='longitude_sections':
            Y,D=np.meshgrid(y,d,indexing='ij');values=np.linspace(0,east[-1],7)
            planes=[(np.full_like(Y,t),Y,D) for t in values]
        elif name=='latitude_sections':
            X,D=np.meshgrid(x,d,indexing='ij');values=np.linspace(0,north[-1],7)
            planes=[(X,np.full_like(X,t),D) for t in values]
        else:
            X,Y=np.meshgrid(x,y,indexing='ij');values=np.linspace(0,depth[-1],7)
            planes=[(X,Y,np.full_like(X,t)) for t in values]
        cuts[name]=values.tolist()
        for X,Y,D in planes:
            v=interp(np.column_stack([X.ravel(),Y.ravel(),D.ravel()])).reshape(X.shape)
            ax.plot_surface(X,Y,D,facecolors=cmap(norm(v)),rcount=X.shape[0],ccount=X.shape[1],
                            shade=False,linewidth=0,antialiased=False)
        annotate_axes(fig,ax,east,north,depth,meta,azim)
        fig.savefig(out/(name+'.png'),dpi=360)
        plt.close(fig)
    fig=plt.figure(figsize=(22/96,340/96),facecolor='white')
    ax=fig.add_axes([0,0,1,1])
    ax.imshow(np.linspace(.1,1,1000).reshape(-1,1),cmap=cmap,norm=norm,aspect='auto',origin='upper')
    ax.axis('off');fig.savefig(out/'colorbar.png',dpi=384);plt.close(fig)
    meta['section_positions_km']=cuts
    (out/'plot_metadata.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2),encoding='utf-8')
    np.savez_compressed(out/'verified_model_grid.npz',east_km=east,north_km=north,depth_km=depth,Vs_km_s=cube)
    print(json.dumps(meta,ensure_ascii=True,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',type=Path,default=Path(__file__).parent/'XJD2_updated.txt')
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args();plot(args.model,args.out)
