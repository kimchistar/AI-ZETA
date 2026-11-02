import sys, os, json, time; sys.path.insert(0,".")
from common import *; from prep import *
from sam_amg import AMG
from skimage import measure
import pandas as pd
def run(case, f0, f1, step, out):
    mk=np.load(f"mask_{case}.npy"); amg=AMG("sam",fp16=True)
    rows=[]; t0=time.time(); frames=list(range(f0,f1,step))
    done=set()
    if os.path.exists(out):
        old=pd.read_csv(out); done=set(old.frame.unique()); rows=old.to_dict("records")
    for n,i in enumerate(frames):
        if i in done: continue
        a=load(case,i); P=preprocess(a,mk)
        lab,sc=amg.run(P["u"],mk,iou_thr=0.5)
        for p in measure.regionprops(lab,intensity_image=P["S"]):
            if p.area<6 or p.intensity_mean<3.0 or sc[p.label]<0.88: continue
            cc=p.coords; res=P["res"][cc[:,0],cc[:,1]]
            rows.append(dict(frame=i,cy=p.centroid[0],cx=p.centroid[1],area=int(p.area),
                ecc=float(p.eccentricity),major=float(p.major_axis_length),minor=float(p.minor_axis_length),
                orient=float(p.orientation),
                I_sum=float(res.sum()),I_peak=float(res.max()),snr_peak=float(P["S"][cc[:,0],cc[:,1]].max()),
                snr_mean=float(p.intensity_mean),score=float(sc[p.label]),sigma=float(P["sigma"]),
                n_sat=int((a[cc[:,0],cc[:,1]]>=SAT[case]).sum())))
        if n%25==0:
            pd.DataFrame(rows).to_csv(out,index=False)
            el=time.time()-t0; print(f"  {case} {n+1}/{len(frames)} frame {i} ({el/(n+1):.1f}s/f, eta {(len(frames)-n-1)*el/(n+1)/60:.0f}min)",flush=True)
    pd.DataFrame(rows).to_csv(out,index=False); print(case,"done",len(rows),"detections",flush=True)
if __name__=="__main__":
    case,f0,f1,step=sys.argv[1],int(sys.argv[2]),int(sys.argv[3]),int(sys.argv[4])
    run(case,f0,f1,step,f"track_det_{case}.csv")
