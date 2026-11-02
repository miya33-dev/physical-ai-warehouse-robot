# SPDX-License-Identifier: AGPL-3.0-only
"""Fine-tune one official YOLO26n checkpoint on generated cube scenes."""
from pathlib import Path
import sys,os,json,hashlib
root=Path(__file__).resolve().parents[1]
for folder in ['.packages','.vision-packages']:
    if (root/folder).exists():sys.path.insert(0,str(root/folder))
os.environ.setdefault('YOLO_CONFIG_DIR',str(root/'.cache/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
os.environ.setdefault('TORCH_HOME',str(root/'models/.torch'))
os.environ['YOLO_AUTOINSTALL']='false'
import argparse
p=argparse.ArgumentParser();p.add_argument('--epochs',type=int,default=30);p.add_argument('--resume',action='store_true');p.add_argument('--dataset',default='cubes_v2');p.add_argument('--name',default='cubes_yolo26n_v2');p.add_argument('--initial',type=Path);a=p.parse_args()
import torch,ultralytics
from ultralytics import YOLO,settings
settings.update({k:False for k in ['sync','clearml','comet','dvc','mlflow','raytune','wandb','tensorboard']})
if not torch.backends.mps.is_available():raise RuntimeError('MPS not available; refusing an unannounced CPU training fallback')
base=root/'models/yolo26n.pt'
checkpoint=root/'runs/detect'/a.name/'weights/last.pt' if a.resume else (a.initial.resolve() if a.initial else base)
model=YOLO(str(checkpoint))
print('Training on MPS:',torch.__version__,ultralytics.__version__,flush=True)
model.train(data=str(root/'datasets'/a.dataset/'data.yaml'),epochs=a.epochs,imgsz=640,batch=8,device='mps',
            workers=0,project=str(root/'runs/detect'),name=a.name,exist_ok=True,
            pretrained=True,optimizer='AdamW',lr0=.001,weight_decay=.0005,
            warmup_epochs=2,patience=10,seed=42,deterministic=True,
            hsv_h=0,hsv_s=.15,hsv_v=.15,degrees=0,translate=.05,scale=.15,
            mosaic=.2,close_mosaic=5,fliplr=.5,flipud=.5,plots=False,amp=False,
            resume=a.resume,save=True,verbose=False)
best=root/'runs/detect'/a.name/'weights/best.pt'
model=YOLO(str(best))
metrics=model.val(data=str(root/'datasets'/a.dataset/'data.yaml'),split='test',device='mps',batch=8,plots=False,
                  project=str(root/'runs/detect'),name=a.name+'_test',exist_ok=True)
card={'architecture':'YOLO26n','base_url':'https://github.com/ultralytics/assets/releases/download/v8.4.0/yolo26n.pt',
      'base_sha256':hashlib.sha256(base.read_bytes()).hexdigest(),'weights_sha256':hashlib.sha256(best.read_bytes()).hexdigest(),
      'weights_path':str(best.relative_to(root)),'initial_checkpoint_path':str(checkpoint.relative_to(root)),'initial_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'license':'AGPL-3.0 (Ultralytics distribution; project source AGPL-3.0-only)',
      'classes':['red_cube','blue_cube'],'ultralytics_version':ultralytics.__version__,'torch_version':torch.__version__,
      'training_device':'mps','training_seed':42,'epochs_requested':a.epochs,'test_metrics':metrics.results_dict,
      'dataset':'datasets/'+a.dataset,'annotation_source':'original procedural MJCF; seed-disjoint splits',
      'limitations':'fixed overhead camera; synthetic tabletop scenes; no claim of real-world accuracy'}
(root/'outputs/detector_training').mkdir(parents=True,exist_ok=True)
(root/'outputs/detector_training/model_card.json').write_text(json.dumps(card,ensure_ascii=False,indent=2))
print(json.dumps(card,ensure_ascii=False,indent=2),flush=True)
