"""Physical wheel-drive checks, including loss of traction and disabled motors."""
from pathlib import Path
import sys,json,math
root=Path(__file__).resolve().parents[1]
if (root/'.packages').exists():sys.path.insert(0,str(root/'.packages'))
sys.path.insert(0,str(root/'src'))
import numpy as np
import mujoco as mj
from warehouse_robot.simulation import Simulation
out=root/'outputs/drive_evaluation';rows=[]
for mode in ['normal','low_traction','motors_disabled']:
    s=Simulation(out/mode,render=False)
    try:
        j=mj.mj_name2id(s.m,mj.mjtObj.mjOBJ_JOINT,'base_free');q=int(s.m.jnt_qposadr[j])
        # Initial test placement in clear floor space; no pose updates while driving.
        s.d.qpos[q+1]=-1.7;mj.mj_forward(s.m,s.d);s.step(.5)
        if mode=='low_traction':
            # Change tire-ground tangential friction only; keep box/table contacts stable.
            for pair,geom in enumerate(s.m.pair_geom2):
                name=mj.mj_id2name(s.m,mj.mjtObj.mjOBJ_GEOM,int(geom))
                if name in ['left_tire','right_tire']:
                    s.m.pair_friction[pair,:2]=.0001
        if mode=='motors_disabled':
            s.m.actuator_gainprm[:2,:]=0;s.m.actuator_biasprm[:2,:]=0
        start=s.base_xy();com_start=s.d.subtree_com[s.base,:2].copy();wheel_start=s.wheel_distance_m.copy()
        s.drive(.25,0,2.);s.stop()
        distance=float(np.linalg.norm(s.base_xy()-start))
        com_distance=float(np.linalg.norm(s.d.subtree_com[s.base,:2]-com_start))
        rotation=(s.wheel_distance_m-wheel_start).tolist()
        if mode=='normal':
            before=s.base_xy();s.turn(math.pi/2)
            yaw_error=abs(math.atan2(math.sin(s.yaw()-math.pi/2),math.cos(s.yaw()-math.pi/2)))
            drift=float(np.linalg.norm(s.base_xy()-before))
            passed=.40<distance<.60 and yaw_error<.03 and drift<.04
        else:
            yaw_error=None;drift=None
            passed=distance<.03
            if mode=='low_traction':
                # Supports still have friction: require strongly reduced propulsion, not zero motion.
                passed=com_distance<.2*rows[0]['center_of_mass_translation_m'] and min(rotation)>.3
        joint_names=[mj.mj_id2name(s.m,mj.mjtObj.mjOBJ_JOINT,j) for j in range(s.m.njnt)]
        passed=bool(passed and 'base_x' not in joint_names and 'base_y' not in joint_names)
        rows.append({'mode':mode,'passed':passed,'translation_m':distance,'center_of_mass_translation_m':com_distance,'wheel_distance_m':rotation,'turn_yaw_error_rad':yaw_error,'turn_drift_m':drift,'floor_contact_steps':s.wheel_contact_steps})
        print(rows[-1],flush=True);s.save_results()
    finally:s.close()
(out/'summary.json').write_text(json.dumps({'cases':rows,'passed':sum(x['passed'] for x in rows),'total':len(rows)},indent=2))
if not all(x['passed'] for x in rows):sys.exit(1)
