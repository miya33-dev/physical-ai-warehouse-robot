"""Test placement generalization with one frozen visual model.
Ground-truth body poses are used exclusively for scoring after RGB detection.
Language uses the baseline classifier to isolate vision and motion.
"""
from pathlib import Path
import sys,json,hashlib
root=Path(__file__).resolve().parents[1]
if (root/'.packages').exists():sys.path.insert(0,str(root/'.packages'))
sys.path.insert(0,str(root/'src'))
import numpy as np
from warehouse_robot.simulation import Simulation
from warehouse_robot.layout import random_layout
out=root/'outputs/layout_evaluation';out.mkdir(parents=True,exist_ok=True)
trainer=Simulation(out/'fixed_training')
try:
    trainer.train_vision(); frozen=trainer.vision
    fingerprint=hashlib.sha256(frozen.means.tobytes()+frozen.invs.tobytes()).hexdigest()
finally:trainer.close()
scenarios=[('seed_'+str(seed),random_layout(seed),seed) for seed in range(8)]
scenarios += [('edge_a',{'red':[-2.10,-.65],'blue':[-1.80,.65]},None),
              ('edge_b',{'red':[-1.80,.65],'blue':[-2.10,-.65]},None)]
rows=[]
for label,layout,seed in scenarios:
    destinations=['shipping','storage'] if seed is not None else ['shipping']
    for color in ['red','blue']:
        for dest in destinations:
            name=label+'_'+color+'_'+dest
            s=Simulation(out/name)
            try:
                s.vision=frozen;s.set_layout(layout,seed=seed)
                # Scoring reference only; never passed to detect/arm/travel.
                idx=0 if color=='red' else 1
                truth=s.d.xpos[s.parcels[idx],:2].copy()
                result=s.execute(('赤' if color=='red' else '青')+'い箱を'+('出荷' if dest=='shipping' else '保管')+'場所へ運んで')
                detected=result.get('detection',{}).get('xy')
                detection_error=float(np.linalg.norm(np.array(detected)-truth)) if detected else None
                passed=bool(result['success'] and result.get('selected_parcel')==idx and detection_error is not None and detection_error<.06)
                rows.append({'case':name,'layout':layout,'passed':passed,'recognition_error_m':detection_error,
                             'placement_error_m':result.get('placement_error_m'),'obstacle_contact_steps':result.get('obstacle_contact_steps'),
                             'error':result.get('error')})
                print(name,passed,'vision_error_m',detection_error,result.get('error',''),flush=True)
            finally:s.close()
unchanged=fingerprint==hashlib.sha256(frozen.means.tobytes()+frozen.invs.tobytes()).hexdigest()
valid_errors=[r['recognition_error_m'] for r in rows if r['recognition_error_m'] is not None]
summary={'passed':sum(r['passed'] for r in rows),'total':len(rows),'visual_model_frozen':unchanged,
         'max_recognition_error_m':max(valid_errors) if valid_errors else None,
         'mean_recognition_error_m':float(np.mean(valid_errors)) if valid_errors else None,
         'scope':'8 reproducible seeds plus 2 edge layouts; fixed camera/table/box height; baseline language to isolate layout handling',
         'cases':rows}
(out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
print('PASS',summary['passed'],'/',summary['total'],'frozen',unchanged,flush=True)
if summary['passed']!=summary['total'] or not unchanged:sys.exit(1)
