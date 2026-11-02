"""Physics integration checks, fresh scene per case, independently scored."""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parent
if (root/'.packages').exists(): sys.path.insert(0,str(root/'.packages'))
sys.path.insert(0,str(root/'src'))
import numpy as np
from warehouse_robot.simulation import Simulation
import mujoco as mj
out=root/'outputs'/'evaluation'; out.mkdir(parents=True,exist_ok=True)
summary=[]
for color in ['red','blue']:
    for dest in ['shipping','storage']:
        name=color+'_'+dest
        s=Simulation(out/name)
        try:
            s.train_vision()
            result=s.execute(('赤' if color=='red' else '青')+'い箱を'+('出荷' if dest=='shipping' else '保管')+'場所へ運んで')
            expected=0 if color=='red' else 1
            passed=result['success'] and result.get('selected_parcel')==expected
            summary.append({'case':name,'passed':bool(passed),'placement_error_m':result.get('placement_error_m'),'obstacle_contact_steps':result.get('obstacle_contact_steps'),'simulated_duration_s':result['simulated_duration_s']})
        finally:s.close()
# Swap actual parcel colors after training: RGB selection must follow appearance.
s=Simulation(out/'swapped_colors')
try:
    s.train_vision()
    ga,gb=s.box_geoms
    c=s.m.geom_rgba[ga].copy();s.m.geom_rgba[ga]=s.m.geom_rgba[gb];s.m.geom_rgba[gb]=c
    result=s.execute('赤い箱を出荷場所へ運んで')
    passed=result['success'] and result.get('selected_parcel')==1
    summary.append({'case':'swapped_colors','passed':bool(passed),'selected_parcel':result.get('selected_parcel'),'placement_error_m':result.get('placement_error_m')})
finally:s.close()
# A missing target must fail before approach; no color-ID shortcut.
s=Simulation(out/'missing_target')
try:
    s.train_vision()
    j=s.m.body_jntadr[s.parcels[0]]; adr=s.m.jnt_qposadr[j]
    s.d.qpos[adr:adr+3]=[3.5,2.5,.59]; mj.mj_forward(s.m,s.d)
    before=s.base_xy()
    result=s.execute('赤い箱を出荷場所へ運んで')
    passed=(not result['success']) and 'not visible' in result.get('error','') and np.linalg.norm(s.base_xy()-before)<.01
    summary.append({'case':'missing_target_stops','passed':bool(passed),'error':result.get('error')})
finally:s.close()
(out/'summary.json').write_text(json.dumps({'cases':summary,'passed':sum(x['passed'] for x in summary),'total':len(summary),'scope':'deterministic fixed scene; no claims about real robots or randomized robustness'},ensure_ascii=False,indent=2))
print(json.dumps(summary,ensure_ascii=False,indent=2))
if not all(x['passed'] for x in summary):sys.exit(1)
