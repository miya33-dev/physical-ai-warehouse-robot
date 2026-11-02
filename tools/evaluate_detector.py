# SPDX-License-Identifier: AGPL-3.0-only
"""Score detections on untouched synthetic test scenes at a fixed 0.5 threshold."""
from pathlib import Path
import sys,json,os
root=Path(__file__).resolve().parents[1]
for folder in ['.packages','.vision-packages']:
    if (root/folder).exists():sys.path.insert(0,str(root/folder))
sys.path.insert(0,str(root/'src'))
from warehouse_robot.detector import CubeDetector
import numpy as np
from PIL import Image,ImageDraw

def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3])
    inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter+1e-9)

def evaluate():
    detector=CubeDetector();manifest=json.loads((root/'datasets/cubes_v2/manifest.json').read_text())
    scenes=[s for s in manifest['scenes'] if s['split']=='test'];tp=fp=fn=0;rows=[];empty_fp=empty_total=0
    out=root/'outputs/detector_evaluation';out.mkdir(parents=True,exist_ok=True)
    gallery=Image.new('RGB',(1280,960))
    for index,s in enumerate(scenes):
        image=Image.open(root/'datasets/cubes_v2'/s['image']).convert('RGB')
        pred=detector.infer(np.array(image));truth=s['boxes'];matched=set();local_tp=0
        for d in sorted(pred,key=lambda x:-x['score']):
            label=0 if d['class']=='red_cube' else 1
            possible=[(iou(d['bbox_xyxy'],t['xyxy']),i) for i,t in enumerate(truth) if i not in matched and t['class']==label]
            score,j=max(possible,default=(0,-1))
            if score>=.5:matched.add(j);tp+=1;local_tp+=1
            else:fp+=1
        missed=len(truth)-len(matched);fn+=missed
        if not truth:empty_total+=1;empty_fp+=bool(pred)
        rows.append({'seed':s['seed'],'ground_truth_count':len(truth),'predictions':pred,'true_positive':local_tp,'false_positive':len(pred)-local_tp,'false_negative':missed})
        if index<4:
            draw=ImageDraw.Draw(image)
            for d in pred:
                color='orange' if d['class']=='red_cube' else 'cyan'
                draw.rectangle(d['bbox_xyxy'],outline=color,width=2)
                x,y=d['bbox_xyxy'][:2];draw.text((x,max(0,y-12)),d['class']+f" {d['score']:.2f}",fill=color)
            gallery.paste(image,((index%2)*640,(index//2)*480))
    result={'weights_sha256':detector.sha256,'split':'test','images':len(scenes),'threshold':.5,'iou_threshold':.5,
            'true_positive':tp,'false_positive':fp,'false_negative':fn,'precision':tp/(tp+fp) if tp+fp else 0,
            'recall':tp/(tp+fn) if tp+fn else 0,'empty_images':empty_total,'empty_images_with_false_detections':empty_fp,
            'scope':'synthetic fixed-camera tabletop scenes with same-color spheres/cylinders/plates; not real-world performance','cases':rows}
    (out/'results.json').write_text(json.dumps(result,ensure_ascii=False,indent=2));gallery.save(out/'predictions.png')
    print(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2))

if __name__=='__main__':evaluate()
