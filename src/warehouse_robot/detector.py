# SPDX-License-Identifier: AGPL-3.0-only
"""Fine-tuned YOLO26n cube detector. Inference receives RGB only."""
from pathlib import Path
import math,hashlib,os

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_WEIGHTS=ROOT/'models/cubes_yolo26n.pt'

def pixel_to_world(x,y,width,height):
    scale=2*(7-.68)*math.tan(math.radians(45)/2)/height
    return [(x-width/2)*scale,(height/2-y)*scale]

def select_target(detections,color,width,height,threshold=.5):
    if color not in ['red','blue']:raise ValueError('unsupported target color')
    candidates=[]
    for d in detections:
        if d['class']!=color+'_cube' or d['score']<threshold:continue
        x1,y1,x2,y2=d['bbox_xyxy']
        if not all(math.isfinite(v) for v in [x1,y1,x2,y2,d['score']]) or not (0<=x1<x2<=width and 0<=y1<y2<=height):continue
        x,y=(x1+x2)/2,(y1+y2)/2
        xy=pixel_to_world(x,y,width,height)
        if -2.25<xy[0]<-1.65 and abs(xy[1])<.8:
            candidates.append({'xy':xy,'pixel_center':[x,y],'bbox_xyxy':[x1,y1,x2,y2],
                               'class':d['class'],'score':d['score']})
    if not candidates:raise ValueError('requested cube not detected in source bay')
    if len(candidates)>1:raise ValueError('multiple requested cubes: specify an unambiguous target')
    return {**candidates[0],'detections':detections,'backend':'yolo26n'}

class CubeDetector:
    name='yolo26n'
    def __init__(self,weights=DEFAULT_WEIGHTS,device='mps',threshold=.5):
        self.weights=Path(weights)
        if not self.weights.is_file():raise ValueError('trained detector missing; run tools/train_detector.py first')
        os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/ultralytics'))
        os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'.cache/matplotlib'))
        os.environ['YOLO_AUTOINSTALL']='false'
        import torch
        from ultralytics import YOLO,settings
        settings.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','raytune','wandb','tensorboard']})
        if device=='mps' and not torch.backends.mps.is_available():raise ValueError('MPS unavailable; select CPU explicitly if required')
        self.model=YOLO(str(self.weights));self.device=device;self.threshold=threshold
        if set(self.model.names.values())!={'red_cube','blue_cube'}:raise ValueError('detector must be trained for red_cube and blue_cube')
        self.sha256=hashlib.sha256(self.weights.read_bytes()).hexdigest()
    def infer(self,rgb):
        from PIL import Image
        result=self.model.predict(source=Image.fromarray(rgb),imgsz=640,device=self.device,
                                  conf=self.threshold,verbose=False)[0]
        boxes=result.boxes
        return [{'class':result.names[int(c)],'score':float(p),'bbox_xyxy':[float(v) for v in b]}
                for b,p,c in zip(boxes.xyxy.cpu().tolist(),boxes.conf.cpu().tolist(),boxes.cls.cpu().tolist())]
    def detect(self,rgb,color):
        h,w=rgb.shape[:2]
        return select_target(self.infer(rgb),color,w,h,self.threshold)
