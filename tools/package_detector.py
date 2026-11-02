# SPDX-License-Identifier: AGPL-3.0-only
"""Create an inference checkpoint with project-relative metadata, not local user paths."""
from pathlib import Path
import sys,os,json,copy,hashlib,zipfile
root=Path(__file__).resolve().parents[1]
for folder in ['.packages','.vision-packages']:
    if (root/folder).exists():sys.path.insert(0,str(root/folder))
os.environ.setdefault('YOLO_CONFIG_DIR',str(root/'.cache/ultralytics'))
os.environ.setdefault('MPLCONFIGDIR',str(root/'.cache/matplotlib'))
import torch,ultralytics
from ultralytics import YOLO,settings

def package_weights():
    settings.update({'sync':False})
    card_path=root/'outputs/detector_training/model_card.json'
    initial_card=json.loads(card_path.read_text())
    source=root/initial_card['weights_path']
    yolo=YOLO(str(source));core=copy.deepcopy(yolo.model).cpu().half()
    def clean(value):
        if isinstance(value,str):
            if str(root) in value:value=value.replace(str(root),'PROJECT_ROOT')
            if value.startswith('/Users/') or value.startswith('/home/'):return Path(value).name
            return value
        if isinstance(value,dict):return {k:clean(v) for k,v in value.items()}
        if isinstance(value,list):return [clean(v) for v in value]
        if isinstance(value,tuple):return tuple(clean(v) for v in value)
        return value
    for module in core.modules():
        for key,value in list(module.__dict__.items()):
            if isinstance(value,(str,dict,list,tuple)):setattr(module,key,clean(value))
    args={'task':'detect','imgsz':640,'model':'yolo26n.pt','data':initial_card['dataset']+'/data.yaml','single_cls':False}
    core.args=args.copy()
    if hasattr(core,'pt_path'):core.pt_path='models/cubes_yolo26n.pt'
    target=root/'models/cubes_yolo26n.pt'
    torch.save({'model':core,'train_args':args,'date':'2026-11-03','version':ultralytics.__version__,
                'license':'AGPL-3.0','docs':'https://docs.ultralytics.com'},target)
    with zipfile.ZipFile(target) as z:
        for name in z.namelist():
            if name.endswith('data.pkl'):
                data=z.read(name)
                if any(x in data for x in [b'/Users/',b'/home/',b'@gmail.com']):raise RuntimeError('private path in packaged checkpoint')
    # Ensure the sanitized checkpoint can still perform model loading.
    loaded=YOLO(str(target))
    assert set(loaded.names.values())=={'red_cube','blue_cube'}
    card_path=root/'outputs/detector_training/model_card.json'
    card=json.loads(card_path.read_text())
    card.update(public_weights_path='models/cubes_yolo26n.pt',public_weights_sha256=hashlib.sha256(target.read_bytes()).hexdigest(),checkpoint_metadata_sanitized=True)
    card_path.write_text(json.dumps(card,ensure_ascii=False,indent=2))
    print('Packaged',target.name,'sha256',card['public_weights_sha256'],flush=True)

if __name__=='__main__':package_weights()
