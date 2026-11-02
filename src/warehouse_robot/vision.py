"""Supervised RGB Gaussian classifier. Labels are used only during training.
Online detection takes RGB only; no simulator IDs or true object poses.
"""
import numpy as np

class PixelModel:
    def fit(self,rgb,labels):
        x=self.features(rgb)
        self.means=[]; self.invs=[]; self.logdets=[]
        for k in range(3):
            a=x[labels==k]
            if len(a)<10: raise ValueError('insufficient class samples')
            cov=np.cov(a.T)+np.eye(x.shape[1])*.0004
            self.means.append(a.mean(0)); self.invs.append(np.linalg.inv(cov))
            self.logdets.append(np.linalg.slogdet(cov)[1])
        self.means=np.array(self.means); self.invs=np.array(self.invs)
        self.logdets=np.array(self.logdets)
        self.n_train=len(labels)
        return self

    @staticmethod
    def features(rgb):
        a=np.asarray(rgb,dtype=float).reshape(-1,3)/255
        return np.column_stack([a,a[:,0]-a[:,1],a[:,2]-a[:,1]])

    def predict(self,rgb):
        x=self.features(rgb)
        scores=[]
        for mu,inv,ld in zip(self.means,self.invs,self.logdets):
            z=x-mu
            scores.append(-.5*(np.einsum('ij,jk,ik->i',z,inv,z)+ld))
        return np.argmax(scores,axis=0)

    def save(self,path):
        np.savez(path,means=self.means,invs=self.invs,logdets=self.logdets,n_train=self.n_train)

    def detect(self,rgb,color):
        h,w=rgb.shape[:2]
        labels=self.predict(rgb).reshape(h,w)
        mask=labels==({'red':1,'blue':2}[color])
        # Known source-bay ROI; calibration is fixed, not SLAM.
        yy,xx=np.where(mask)
        scale=2*(7-.68)*np.tan(np.deg2rad(45)/2)/h
        wx=(xx-w/2+.5)*scale; wy=(h/2-yy-.5)*scale
        keep=(wx>-2.25)&(wx<-1.65)&(np.abs(wy)<.8)
        if keep.sum()<30: raise ValueError('requested parcel not visible in source bay')
        return {'xy':[float(np.median(wx[keep])),float(np.median(wy[keep]))],
                'pixels':int(keep.sum()),'pixel_center':[float(np.median(xx[keep])),float(np.median(yy[keep]))]}
