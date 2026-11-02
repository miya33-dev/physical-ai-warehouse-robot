"""Grasp physics controls: normal friction, open fingers, low pad friction.
Known target coordinates here isolate grasp mechanics from perception.
"""
from pathlib import Path
import sys,json
root=Path(__file__).resolve().parents[1]
if (root/'.packages').exists():sys.path.insert(0,str(root/'.packages'))
sys.path.insert(0,str(root/'src'))
from warehouse_robot.simulation import Simulation
out=root/'outputs/gripper_evaluation';rows=[]
for mode in ['normal','open_fingers','low_friction']:
    s=Simulation(out/mode,render=False)
    try:
        target=[-1.95,-.4]
        s.travel((-2.6,-.4));s.arm(target,.95);s.arm(target,.69,1.8)
        if mode=='low_friction':
            for geom in s.finger_geoms+s.box_geoms:s.m.geom_friction[geom,0]=.0001
        contact_error=None
        if mode!='open_fingers':
            try:s.grasp()
            except RuntimeError as exc:contact_error=str(exc)
        forces=s.grip_contacts().tolist()
        s.arm(target,1.02,1.8);lift=float(s.d.xpos[s.parcels[0],2]);s.step(10.)
        hold=float(s.d.xpos[s.parcels[0],2])
        if mode=='normal':
            passed=contact_error is None and lift>.84 and hold>.84
        else:passed=hold<.65
        # If the box stayed on the table, open before lowering to avoid hitting it.
        if mode!='normal':s.gripper(.15)
        s.arm(target,.69,1.8);s.gripper(.15);s.step(.8)
        release=float(s.d.xpos[s.parcels[0],2])
        passed=bool(passed and s.m.neq==0 and abs(release-.59)<.04)
        rows.append({'mode':mode,'passed':passed,'equality_constraints':int(s.m.neq),'closing_forces_n':forces,
                     'lifted_center_height_m':lift,'height_after_10s_m':hold,'released_center_height_m':release,'grasp_error':contact_error})
        print(rows[-1],flush=True);s.save_results()
    finally:s.close()
(out/'summary.json').write_text(json.dumps({'passed':sum(r['passed'] for r in rows),'total':len(rows),'cases':rows},ensure_ascii=False,indent=2))
if not all(r['passed'] for r in rows):sys.exit(1)
