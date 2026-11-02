import numpy as np, torch, torch.nn.functional as F
from transformers import SamModel, SamProcessor, Sam2Model, Sam2Processor
DEV="cuda"
class AMG:
    def __init__(self,kind="sam",fp16=False):
        self.kind=kind; self.fp16=fp16
        if kind=="sam":
            self.model=SamModel.from_pretrained("facebook/sam-vit-huge").to(DEV).eval()
            self.proc=SamProcessor.from_pretrained("facebook/sam-vit-huge")
        else:
            self.model=Sam2Model.from_pretrained("facebook/sam2.1-hiera-large").to(DEV).eval()
            self.proc=Sam2Processor.from_pretrained("facebook/sam2.1-hiera-large")
    def _pts(self,P,H,W):
        P=P.astype(np.float32)
        if self.kind=="sam":
            s=1024/max(H,W); return P*s
        return P/np.array([W,H],np.float32)*1024
    def _up(self,low,H,W):
        if self.kind=="sam":
            s=1024/max(H,W); rh,rw=int(H*s+.5),int(W*s+.5)
            x=F.interpolate(low,(1024,1024),mode="bilinear",align_corners=False)[...,:rh,:rw]
            return F.interpolate(x,(H,W),mode="bilinear",align_corners=False)
        return F.interpolate(low,(H,W),mode="bilinear",align_corners=False)
    @torch.no_grad()
    def run(self,u,mk,spacing=5,iou_thr=0.70,stab_thr=0.85,max_area_frac=0.05,nms_iou=0.5,batch=128):
        ys,xs=np.where(mk); y0,y1,x0,x1=ys.min(),ys.max()+1,xs.min(),xs.max()+1
        crop=(np.clip(u[y0:y1,x0:x1],0,1)*255).astype(np.uint8)
        img=np.repeat(crop[...,None],3,2); H,W=crop.shape
        gy,gx=np.mgrid[spacing//2:H:spacing, spacing//2:W:spacing]
        pts=np.stack([gx.ravel(),gy.ravel()],1)
        pts=pts[mk[y0:y1,x0:x1][pts[:,1],pts[:,0]]]
        inp=self.proc(images=img,return_tensors="pt").to(DEV)
        with torch.autocast("cuda",dtype=torch.float16,enabled=self.fp16):
            emb=self.model.get_image_embeddings(inp["pixel_values"])
        maxA=max_area_frac*mk.sum(); cand=[]
        for b in range(0,len(pts),batch):
            P=pts[b:b+batch]
            ip=torch.from_numpy(self._pts(P,H,W))[None,:,None,:].to(DEV)
            il=torch.ones(1,len(P),1,dtype=torch.long,device=DEV)
            with torch.autocast("cuda",dtype=torch.float16,enabled=self.fp16):
                out=self.model(image_embeddings=emb,input_points=ip,input_labels=il,multimask_output=True)
            low=out.pred_masks[0].float()
            iou=out.iou_scores[0].float()
            m=self._up(low,H,W)
            area=(m>0).flatten(2).sum(-1); stab=(m>1).flatten(2).sum(-1).float()/(m>-1).flatten(2).sum(-1).clamp(min=1).float()
            own=m[torch.arange(len(P)),:,torch.from_numpy(P[:,1]).to(DEV),torch.from_numpy(P[:,0]).to(DEV)]>0
            ok=(area>=6)&(area<=maxA)&(iou>=iou_thr)&(stab>=stab_thr)&own
            sc=torch.where(ok,iou,torch.full_like(iou,-1)); bs,bk=sc.max(1)
            for n in torch.nonzero(bs>0).flatten().tolist():
                k=int(bk[n]); cand.append((float(bs[n]),float(stab[n,k]),(m[n,k]>0).cpu().numpy()))
        cand.sort(key=lambda t:-t[0]); occ=np.zeros((H,W),np.int32); keep=[]
        for sc,st,mm in cand:
            A=mm.sum()
            if (occ[mm]>0).sum()/A>nms_iou: continue
            mm=mm&(occ==0)
            if mm.sum()<6: continue
            keep.append(sc); occ[mm]=len(keep)
        lab=np.zeros(mk.shape,np.int32); lab[y0:y1,x0:x1]=occ; lab[~mk]=0
        score=np.zeros(len(keep)+1); score[1:]=keep
        return lab,score
