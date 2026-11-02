# SPDX-License-Identifier: AGPL-3.0-only
"""End-to-end perception/grasp/drive checks with one frozen YOLO checkpoint."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
for folder in ['.packages','.vision-packages']:
    if (root/folder).exists():sys.path.insert(0,str(root/folder))
sys.path.insert(0,str(root/'src'))
import numpy as np
from warehouse_robot.detector import CubeDetector
from warehouse_robot.simulation import Simulation
from warehouse_robot.layout import random_layout

def evaluate():
    detector=CubeDetector();out=root/'outputs/yolo_robot_evaluation';rows=[]
    cases=[(seed,color,destination,False) for seed in [200,201,202,203] for color in ['red','blue'] for destination in ['shipping','storage']]
    cases += [(0,c,d,True) for c,d in [('red','shipping'),('blue','storage')]]
    for seed,color,destination,clutter in cases:
        name=f'seed_{seed}_{color}_{destination}'+('_clutter' if clutter else '')
        s=Simulation(out/name)
        try:
            s.vision=detector;s.set_layout(random_layout(seed),seed=seed)
            if clutter:s.enable_distractors()
            idx=0 if color=='red' else 1;truth=s.d.xpos[s.parcels[idx],:2].copy()
            result=s.execute(('赤' if color=='red' else '青')+'い箱を'+('出荷' if destination=='shipping' else '保管')+'場所へ運んで')
            xy=result.get('detection',{}).get('xy');error=float(np.linalg.norm(np.array(xy)-truth)) if xy else None
            passed=bool(result['success'] and result.get('selected_parcel')==idx and error is not None and error<.06)
            rows.append({'case':name,'passed':passed,'recognition_error_m':error,'placement_error_m':result.get('placement_error_m'),'error':result.get('error')})
            print(name,passed,'recognition_error_m',error,result.get('error',''),flush=True)
        finally:s.close()
    out.mkdir(parents=True,exist_ok=True)
    summary={'passed':sum(r['passed'] for r in rows),'total':len(rows),'weights_sha256':detector.sha256,
             'scope':'four new layout seeds and same-color clutter; Gaussian classifier is not used; language baseline isolates detector/motion',
             'cases':rows}
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print('PASS',summary['passed'],'/',summary['total'],flush=True)
    if summary['passed']!=summary['total']:sys.exit(1)

if __name__=='__main__':evaluate()
