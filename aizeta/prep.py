import numpy as np, cv2
from scipy import ndimage as ndi
from skimage import measure, feature, filters, morphology, segmentation
SAT={"31_A003":45853,"2_TRIP1":45853,"8_TRIP2":45853,"21_V19_1A":65535,"26_V19_5A":65535}
SMAX=40.0
MIN_AREA=6

def background(a):
    small=cv2.resize(a,(400,400),interpolation=cv2.INTER_AREA)
    k=cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(25,25))
    bg=cv2.morphologyEx(small,cv2.MORPH_OPEN,k)
    bg=cv2.GaussianBlur(bg,(0,0),3)
    return cv2.resize(bg,(800,800),interpolation=cv2.INTER_LINEAR)

def preprocess(a,mk):
    bg=background(a); r=a-bg
    rv=r[mk]; sig=1.4826*np.median(np.abs(rv-np.median(rv)))
    S=(r/sig)*mk
    u=np.clip(np.arcsinh(np.clip(S,0,None)/3)/np.arcsinh(SMAX/3),0,1)
    return dict(bg=bg,res=r,sigma=sig,S=S,u=u)

def regions_from_label(lab,a,S,case,method,score=None):
    out=[]
    for p in measure.regionprops(lab,intensity_image=a):
        if p.area<MIN_AREA: continue
        cc=p.coords
        out.append(dict(method=method,cy=p.centroid[0],cx=p.centroid[1],area=int(p.area),
            ecc=float(p.eccentricity),major=float(p.major_axis_length),minor=float(p.minor_axis_length),
            peak=float(p.intensity_max),snr_peak=float(S[cc[:,0],cc[:,1]].max()),
            n_sat=int((a[cc[:,0],cc[:,1]]>=SAT[case]).sum()),
            bbox=p.bbox,score=float(score[p.label]) if score is not None else np.nan,label=int(p.label)))
    return out

def det_global_raw(a,mk,case,k=3.0):
    v=a[mk]; med=np.median(v); rs=1.4826*np.median(np.abs(v-med))
    t=med+k*rs
    return measure.label((a>t)&mk),t

def det_bgsub_thresh(P,mk,k=5.0):
    return measure.label((P["S"]>k)&mk)

def det_bgsub_watershed(P,mk,k=5.0):
    b=(P["S"]>k)&mk
    sm=ndi.gaussian_filter(P["S"],1.0)
    pk=feature.peak_local_max(sm,min_distance=4,threshold_abs=k,labels=measure.label(b))
    mkr=np.zeros_like(b,int); mkr[tuple(pk.T)]=np.arange(1,len(pk)+1)
    return segmentation.watershed(-sm,mkr,mask=b)

def det_canny(P,mk,sigma=1.5,snr_min=3.0):
    e=feature.canny(P["u"],sigma=sigma)&mk
    e=morphology.binary_closing(e,morphology.disk(1))
    f=ndi.binary_fill_holes(e)
    lab=measure.label(f)
    keep=np.zeros(lab.max()+1,bool)
    for p in measure.regionprops(lab,intensity_image=P["S"]):
        keep[p.label]=p.intensity_mean>snr_min
    lab[~keep[lab]]=0
    return measure.label(lab>0)

def det_log(P,mk,thr=0.05):
    bl=feature.blob_log(P["u"],min_sigma=1.5,max_sigma=10,num_sigma=10,threshold=thr)
    lab=np.zeros((800,800),int); yy,xx=np.mgrid[:800,:800]
    for n,(y,x,s) in enumerate(bl,1):
        y,x=int(y),int(x)
        if not mk[y,x]: continue
        r=int(np.ceil(s*np.sqrt(2)))+1
        y0,y1,x0,x1=max(y-r,0),min(y+r+1,800),max(x-r,0),min(x+r+1,800)
        disk=(yy[y0:y1,x0:x1]-y)**2+(xx[y0:y1,x0:x1]-x)**2<=r*r
        sub=lab[y0:y1,x0:x1]; sub[disk&(sub==0)]=n
    return lab
