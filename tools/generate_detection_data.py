# SPDX-License-Identifier: AGPL-3.0-only
"""Generate RGB/visible-box labels from project-authored MuJoCo scenes.
Segmentation IDs are annotation data only. No lecture or external images are used.
"""
from pathlib import Path
import sys,json,xml.etree.ElementTree as ET
root=Path(__file__).resolve().parents[1]
if (root/'.packages').exists():sys.path.insert(0,str(root/'.packages'))
import numpy as np
import mujoco as mj
from PIL import Image

COLORS=[np.array([.90,.08,.05]),np.array([.06,.20,.90])]

def build_scene():
    xml=ET.parse(root/'assets/warehouse.xml');world=xml.getroot().find('worldbody')
    for i in range(4):
        ET.SubElement(world,'geom',name=f'syn_cube_{i}',type='box',pos='0 0 -10',size='.09 .09 .09',rgba='.9 .1 .1 1',contype='0',conaffinity='0')
    for i,shape in enumerate(['sphere','sphere','cylinder','cylinder','box','box']):
        size='.09' if shape=='sphere' else ('.09 .09' if shape=='cylinder' else '.09 .09 .015')
        ET.SubElement(world,'geom',name=f'syn_distractor_{i}',type=shape,pos='0 0 -10',size=size,rgba='.1 .2 .9 1',contype='0',conaffinity='0')
    m=mj.MjModel.from_xml_string(ET.tostring(xml.getroot(),encoding='unicode'));d=mj.MjData(m)
    for name,value in [('shoulder',-.9),('elbow',2.2),('wrist',-1.3),('left_finger',.15),('right_finger',.15)]:
        j=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,name);d.qpos[m.jnt_qposadr[j]]=value
    return m,d

def generate(counts=(512,96,96),variant='v2'):
    dest=root/('datasets/cubes_'+variant);dest.mkdir(parents=True,exist_ok=True)
    m,d=build_scene();r=mj.Renderer(m,height=480,width=640)
    original_rgba=m.geom_rgba.copy();original_size=m.geom_size.copy();original_mat=m.mat_rgba.copy()
    original_light=m.light_diffuse.copy();original_light_pos=m.light_pos.copy()
    cubes=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['box_a','box_b']+[f'syn_cube_{i}' for i in range(4)]]
    extras=[mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,f'syn_distractor_{i}') for i in range(6)]
    parcel_q=[]
    for n in ['parcel_a','parcel_b']:
        b=mj.mj_name2id(m,mj.mjtObj.mjOBJ_BODY,n);parcel_q.append(int(m.jnt_qposadr[m.body_jntadr[b]]))
    base=mj.mj_name2id(m,mj.mjtObj.mjOBJ_JOINT,'base_free');base_q=int(m.jnt_qposadr[base])
    metadata=[]
    try:
        for split,count,offset in zip(['train','val','test'],counts,[10000,20000,30000] if variant=='v1' else [40000,50000,60000]):
            (dest/'images'/split).mkdir(parents=True,exist_ok=True);(dest/'labels'/split).mkdir(parents=True,exist_ok=True)
            for index in range(count):
                seed=offset+index;rng=np.random.default_rng(seed)
                m.geom_rgba[:]=original_rgba;m.geom_size[:]=original_size;m.mat_rgba[:]=original_mat
                for q in parcel_q:d.qpos[q:q+3]=[10,10,-5]
                for g in cubes[2:]+extras:m.geom_pos[g]=[10,10,-5]
                n_cubes=int(rng.choice([0,1,2,3,4],p=[.15,.15,.4,.2,.1]));n_other=int(rng.integers(1,7))
                n_other=min(n_other,10-n_cubes)
                continuous=variant=='v2' and rng.random()<.65
                if continuous:
                    n_cubes=min(n_cubes,2);n_other=min(n_other,3)
                positions=[(x,y) for x in [-2.075,-1.825] for y in [-.62,-.31,0,.31,.62]]
                rng.shuffle(positions);rng.shuffle(cubes)
                if continuous:
                    total=n_cubes+n_other
                    for _ in range(100000):
                        ys=np.sort(rng.uniform(-.65,.65,total))
                        if total<2 or np.min(np.diff(ys))>.22:
                            positions=list(zip(rng.uniform(-2.10,-1.80,total),ys))
                            rng.shuffle(positions)
                            break
                    else:raise RuntimeError('could not place continuous objects')
                # Original free boxes use joint qpos; extra boxes use model geom positions.
                labels={};truth=[]
                chosen=cubes[:n_cubes]+list(rng.choice(extras,n_other,replace=False))
                for slot,g in enumerate(chosen):
                    x,y=np.array(positions[slot])+(rng.uniform(-.018,.018,2) if not continuous else 0)
                    cls=int(rng.integers(0,2));brightness=rng.uniform(.6,1.1)
                    m.geom_rgba[g,:3]=np.clip(COLORS[cls]*brightness+rng.uniform(-.025,.025,3),.015,.98)
                    cube=g in cubes
                    half=float(rng.uniform(.075,.105)) if cube else float(m.geom_size[g,0])
                    if cube:m.geom_size[g,:]=half
                    kind=int(m.geom_type[g])
                    h=half if cube or kind==int(mj.mjtGeom.mjGEOM_SPHERE) else float(m.geom_size[g,1] if kind==int(mj.mjtGeom.mjGEOM_CYLINDER) else m.geom_size[g,2])
                    if g==mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'box_a'):d.qpos[parcel_q[0]:parcel_q[0]+3]=[x,y,.5+h]
                    elif g==mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,'box_b'):d.qpos[parcel_q[1]:parcel_q[1]+3]=[x,y,.5+h]
                    else:m.geom_pos[g]=[x,y,.5+h]
                    if cube:labels[g]=cls
                    truth.append({'geom':mj.mj_id2name(m,mj.mjtObj.mjOBJ_GEOM,g),'cube':cube,'color':cls,'xy':[float(x),float(y)]})
                # Vary material/lighting, while retaining the same calibrated camera.
                for name in ['source','shipping','storage']:
                    g=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name);m.geom_rgba[g,:3]=rng.uniform(.25,.8,3)
                m.mat_rgba[:,:3]=rng.uniform(.5,1.,(m.nmat,3))
                m.light_diffuse[:]=rng.uniform(.5,1.,(m.nlight,3));m.light_pos[0,:2]=rng.uniform([-2,-3],[2,0])
                if variant=='v2' and rng.random()<.40:
                    for name in ['source','shipping','storage']:
                        g=mj.mj_name2id(m,mj.mjtObj.mjOBJ_GEOM,name);m.geom_rgba[g]=original_rgba[g]
                    m.mat_rgba[:]=original_mat;m.light_diffuse[:]=original_light;m.light_pos[:]=original_light_pos
                d.qpos[base_q+1]=rng.uniform(-1.6,1.6)
                mj.mj_forward(m,d)
                r.disable_segmentation_rendering();r.update_scene(d,camera='overhead');rgb=r.render().copy()
                r.enable_segmentation_rendering();r.update_scene(d,camera='overhead');seg=r.render().copy();r.disable_segmentation_rendering()
                boxes=[];lines=[]
                for g,cls in labels.items():
                    yy,xx=np.where((seg[:,:,0]==g)&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM)))
                    if len(xx)<25:continue
                    box=[int(xx.min()),int(yy.min()),int(xx.max()+1),int(yy.max()+1)]
                    x1,y1,x2,y2=box
                    lines.append(f'{cls} {(x1+x2)/1280:.7f} {(y1+y2)/960:.7f} {(x2-x1)/640:.7f} {(y2-y1)/480:.7f}')
                    boxes.append({'class':cls,'xyxy':box})
                name=f'{seed:06d}'
                Image.fromarray(rgb).save(dest/'images'/split/(name+'.png'))
                (dest/'labels'/split/(name+'.txt')).write_text('\n'.join(lines)+'\n' if lines else '')
                metadata.append({'split':split,'seed':seed,'image':f'images/{split}/{name}.png','boxes':boxes,'objects':truth})
                if index%64==0:print(split,index,'/',count,flush=True)
        (dest/'manifest.json').write_text(json.dumps({'generator':'tools/generate_detection_data.py','classes':['red_cube','blue_cube'],'variant':variant,'counts':dict(zip(['train','val','test'],counts)),'annotation':'visible bounding boxes from simulator masks; inference uses RGB only','scenes':metadata},indent=2))
        (dest/'data.yaml').write_text(f'path: {dest}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: red_cube\n  1: blue_cube\n')
    finally:r.close()

if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--variant',choices=['v1','v2'],default='v2');a=p.parse_args()
    generate(variant=a.variant)
