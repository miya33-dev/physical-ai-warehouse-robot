"""MuJoCo SCARA mobile manipulator, wheel-driven differential base and contact gripper.
Known map and simulator pose feedback; parallel fingers with contact/friction grasping.
"""
from pathlib import Path
import json
import math
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont,ImageOps
from .planning import plan
from .drive import wheel_speeds,wrap,corners,WHEEL_RADIUS
from .language import IntentModel
from .vision import PixelModel
from .layout import validate_layout

ROOT=Path(__file__).resolve().parents[1]
if ROOT.name=='src': ROOT=ROOT.parent
DESTINATIONS={'shipping':(1.9,-1.25),'storage':(1.9,1.25)}

class Simulation:
    def __init__(self,out,render=True,video=False,language_model=None):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.m=mj.MjModel.from_xml_path(str(ROOT/'assets/warehouse.xml'))
        self.d=mj.MjData(self.m)
        self.r=mj.Renderer(self.m,height=480,width=640) if render else None
        self.cam=mj.MjvCamera(); self.cam.lookat[:]=[0,0,.3]; self.cam.distance=8.8; self.cam.azimuth=125; self.cam.elevation=-43
        self.joints=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_JOINT,n) for n in ['shoulder','elbow','lift','wrist','left_finger','right_finger']]
        self.qids=[int(self.m.jnt_qposadr[j]) for j in self.joints]
        self.goal=np.array([0,0,-.9,2.2,0.,-1.3,.15,.15])
        self.d.qpos[self.qids]=self.goal[2:]; self.d.ctrl[:]=self.goal
        self.base=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'base')
        self.wheel_joints=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_JOINT,n) for n in ['left_wheel','right_wheel']]
        self.wheel_qids=[int(self.m.jnt_qposadr[j]) for j in self.wheel_joints]
        self.wheel_distance_m=np.zeros(2); self.wheel_contact_steps=0; self.navigation=[]; self.furniture_contacts=0; self.video_speed=1
        self.wheel_geoms={mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['left_tire','right_tire']}
        self.furniture_geoms={mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['source','shipping','storage']}
        self.tip=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_SITE,'tip')
        self.tool=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'tool')
        self.finger_geoms=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['left_pad','right_pad']]
        self.grasp_events=[]; self.monitor_grasp=False
        self.obstacle=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'obstacle')
        self.robot_geoms=set(range(mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'chassis'),self.m.ngeom))
        self.parcels=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,n) for n in ['parcel_a','parcel_b']]
        self.box_geoms=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['box_a','box_b']]
        self.contact_pairs=set(); self.collisions=0; self.held=None; self.stage='INITIALIZE'; self.command=''; self.intent=None
        self.frames=0; self.records=[]; self.trail=[]; self.vision_frame=None; self.detection=None
        self.layout_metadata={'mode':'fixed'};self.distractor_metadata=[]
        self.lang=language_model if language_model is not None else IntentModel(); self.writer=None
        if video: self.start_video()
        mj.mj_forward(self.m,self.d)
        self.step(.6)

    def start_video(self,speed=1):
        if self.writer is not None:
            raise ValueError('video recording is already active')
        if speed not in [1,2]:raise ValueError('video speed must be 1 or 2')
        self.video_speed=speed
        import imageio.v2 as iio
        self.writer=iio.get_writer(str(self.out/'demo.mp4'),fps=15,codec='libx264',quality=8,macro_block_size=16)

    def set_layout(self,layout,seed=None):
        """Reset parcel initial poses before inference; requires frozen vision model."""
        if not hasattr(self,'vision'):
            raise ValueError('train or load the vision model before setting evaluation layout')
        if self.records or self.held is not None:
            raise ValueError('layout can only change before any delivery starts')
        layout=validate_layout(layout)
        for color,body in zip(['red','blue'],self.parcels):
            joint=int(self.m.body_jntadr[body])
            q=int(self.m.jnt_qposadr[joint]); v=int(self.m.jnt_dofadr[joint])
            self.d.qpos[q:q+3]=[*layout[color],.59]
            self.d.qpos[q+3:q+7]=[1,0,0,0]
            self.d.qvel[v:v+6]=0
        self.layout_metadata={'mode':'random' if seed is not None else 'custom','seed':seed,'requested_xy':layout}
        mj.mj_forward(self.m,self.d)
        self.step(.4)
        self.vision_frame=self.rgb(); self.detection=None
        Image.fromarray(self.vision_frame).save(self.out/'layout_camera.png')
        self.save_results()

    def enable_distractors(self):
        """Initial scene setup only; keep the gripper approach clear."""
        if self.records or self.held is not None:raise ValueError('set distractors before delivery')
        occupied=[float(self.d.xpos[b,1]) for b in self.parcels]
        for i,name in enumerate(['demo_red_sphere','demo_blue_cylinder','demo_red_plate']):
            candidates=[y for y in np.linspace(-.67,.67,15) if min(abs(y-v) for v in occupied)>.26]
            if not candidates:break
            y=max(candidates,key=lambda x:min(abs(x-v) for v in occupied));occupied.append(y)
            g=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,name)
            h=.012 if 'plate' in name else .065
            self.m.geom_pos[g]=[-2.03 if i%2==0 else -1.87,y,.5+h]
            self.distractor_metadata.append({'name':name,'position':self.m.geom_pos[g].tolist()})
        mj.mj_forward(self.m,self.d);self.vision_frame=self.rgb()

    def close(self):
        if self.writer: self.writer.close(); self.writer=None
        if self.r: self.r.close(); self.r=None

    def base_xy(self): return self.d.xpos[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'base'),:2].copy()

    def step(self,seconds):
        for i in range(round(seconds/self.m.opt.timestep)):
            self.d.ctrl[:]=self.goal
            previous=self.d.qpos[self.wheel_qids].copy()
            mj.mj_step(self.m,self.d)
            if self.monitor_grasp and self.held is not None and self.d.xpos[self.parcels[self.held],2]<.75:
                self.grasp_events.append({'phase':self.stage,'event':'dropped','time_s':float(self.d.time),'parcel_position':self.d.xpos[self.parcels[self.held]].tolist(),'forces':self.grip_contacts().tolist()})
                self.monitor_grasp=False;self.goal[:2]=0
                raise RuntimeError('parcel dropped during '+self.stage)
            self.wheel_distance_m+=np.abs(self.d.qpos[self.wheel_qids]-previous)*WHEEL_RADIUS
            if any((int(c.geom1) in self.wheel_geoms or int(c.geom2) in self.wheel_geoms) and (int(c.geom1)==0 or int(c.geom2)==0) for c in self.d.contact):
                self.wheel_contact_steps+=1
            for c in self.d.contact:
                a,b=int(c.geom1),int(c.geom2)
                if (a in self.furniture_geoms and b in self.robot_geoms) or (b in self.furniture_geoms and a in self.robot_geoms):
                    if c.dist<-.002:self.furniture_contacts+=1
                if self.obstacle in (a,b) and (a in self.robot_geoms or b in self.robot_geoms or a in self.box_geoms or b in self.box_geoms):
                    if c.dist<-.001:
                        self.collisions+=1; self.contact_pairs.add(tuple(sorted((a,b))))
            if self.writer and round(self.d.time/self.m.opt.timestep)%(17*self.video_speed)==0:
                self.frame()

    def move(self,goal,seconds=1.5):
        start=self.goal.copy(); goal=np.array(goal,float)
        n=max(1,round(seconds/.04))
        for i in range(1,n+1):
            t=i/n; t=t*t*(3-2*t)
            self.goal=start+(goal-start)*t
            self.step(.04)
        self.goal=goal; self.step(.25)

    def arm(self,xy,z,seconds=1.5):
        rotation=self.d.xmat[self.base].reshape(3,3)
        x,y=rotation[:2,:2].T@(np.array(xy)-self.base_xy())
        cc=(x*x+y*y-2*.45**2)/(2*.45**2)
        if not -1<=cc<=1: raise ValueError('target outside arm reach')
        e=math.acos(cc)
        a=math.atan2(y,x)-math.atan2(.45*math.sin(e),.45+.45*math.cos(e))
        q=self.goal.copy(); q[2:5]=[a,e,z-(self.d.xpos[self.base,2]+.905)]
        q[5]=wrap(-self.yaw()-a-e)
        self.move(q,seconds)
        err=float(np.linalg.norm(self.d.site_xpos[self.tip]-np.array([xy[0],xy[1],z])))
        if err>.04: raise RuntimeError(f'arm tracking error {err:.3f} m')

    def yaw(self):
        rot=self.d.xmat[self.base].reshape(3,3)
        return math.atan2(rot[1,0],rot[0,0])

    def drive(self,v,w,dt=.04):
        self.goal[:2]=wheel_speeds(v,w)
        self.step(dt)

    def stop(self):
        self.goal[:2]=0
        self.step(.4)

    def turn(self,heading):
        for _ in range(450):
            error=wrap(heading-self.yaw())
            if abs(error)<.012:break
            self.drive(0,float(np.clip(2.5*error,-1.,1.)))
        else:
            self.stop();raise RuntimeError('wheel-driven heading timeout')
        self.stop()

    def travel(self,xy):
        start=self.base_xy(); route=plan(start,xy)
        self.trail.extend(route)
        points=corners(route)
        if np.linalg.norm(np.array(points[-1])-xy)>.005:points.append(tuple(xy))
        before=self.wheel_distance_m.copy()
        for target in points[1:]:
            target=np.array(target)
            offset=target-self.base_xy()
            if np.linalg.norm(offset)<.025:continue
            self.turn(math.atan2(offset[1],offset[0]))
            for _ in range(1800):
                offset=target-self.base_xy(); distance=float(np.linalg.norm(offset))
                if distance<.022:break
                error=wrap(math.atan2(offset[1],offset[0])-self.yaw())
                speed=min(.34,.9*distance) if abs(error)<.35 else 0.
                self.drive(speed,float(np.clip(2.8*error,-1.,1.)))
            else:
                self.stop();raise RuntimeError('wheel-driven waypoint timeout')
            self.stop()
        self.turn(0.)
        error=float(np.linalg.norm(self.base_xy()-xy))
        self.navigation.append({'start_xy':start.tolist(),'goal_xy':list(xy),'final_xy':self.base_xy().tolist(),
                                'position_error_m':error,'wheel_distance_m':(self.wheel_distance_m-before).tolist()})
        if error>.06:raise RuntimeError('wheel-driven docking error')
        return route

    def rgb(self):
        self.r.disable_segmentation_rendering()
        self.r.update_scene(self.d,camera='overhead')
        return self.r.render().copy()

    def train_vision(self):
        rng=np.random.default_rng(42); xs=[]; ys=[]
        original=self.m.geom_rgba.copy()
        for i in range(12):
            # Simulator masks are training annotations ONLY, not inference input.
            for k,g in enumerate(self.box_geoms):
                nominal=np.array([.90,.08,.05]) if k==0 else np.array([.06,.20,.90])
                self.m.geom_rgba[g,:3]=np.clip(nominal*rng.uniform(.70,1.05)+rng.uniform(-.025,.025,3),.02,.98)
            rgb=self.rgb()
            self.r.enable_segmentation_rendering(); self.r.update_scene(self.d,camera='overhead'); seg=self.r.render().copy()
            self.r.disable_segmentation_rendering()
            labels=np.zeros(seg.shape[:2],int)
            for k,g in enumerate(self.box_geoms): labels[(seg[:,:,0]==g)&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))]=k+1
            for k in range(3):
                ii=np.flatnonzero(labels.ravel()==k)
                if len(ii)==0: raise RuntimeError('parcel missing from training camera')
                ii=rng.choice(ii,min(len(ii),800),replace=False)
                samples=rgb.reshape(-1,3)[ii].astype(float)+rng.normal(0,3,(len(ii),3))
                xs.append(np.clip(samples,0,255)); ys.append(np.full(len(ii),k))
        self.m.geom_rgba[:]=original
        self.vision=PixelModel().fit(np.concatenate(xs),np.concatenate(ys))
        self.vision.save(self.out/'vision_model.npz')
        rgb=self.rgb(); self.vision_frame=rgb
        # A held-out rendering at the nominal colors. Report per-class accuracy.
        self.r.enable_segmentation_rendering(); self.r.update_scene(self.d,camera='overhead'); seg=self.r.render().copy(); self.r.disable_segmentation_rendering()
        truth=np.zeros(seg.shape[:2],int)
        for k,g in enumerate(self.box_geoms): truth[(seg[:,:,0]==g)&(seg[:,:,1]==int(mj.mjtObj.mjOBJ_GEOM))]=k+1
        pred=self.vision.predict(rgb).reshape(truth.shape)
        acc={name:float((pred[truth==k]==k).mean()) for k,name in enumerate(['background','red','blue'])}
        (self.out/'training_metrics.json').write_text(json.dumps({'seed':42,'training_samples':self.vision.n_train,'language_training_samples':getattr(self.lang,'n_train',None),'nominal_heldout_pixel_accuracy':acc,'note':'synthetic fixed-camera scene only; not real-world validation'},indent=2))
        Image.fromarray(rgb).save(self.out/'camera.png')
        return acc

    def gripper_rgb(self):
        camera=mj.MjvCamera();camera.lookat[:]=self.d.xpos[self.tool]+[0,0,-.06]
        camera.distance=.70;camera.azimuth=135;camera.elevation=-25
        self.r.disable_segmentation_rendering();self.r.update_scene(self.d,camera=camera)
        return self.r.render().copy()

    def grip_contacts(self):
        forces=np.zeros((2,2))
        for index,c in enumerate(self.d.contact):
            pair={int(c.geom1),int(c.geom2)}
            for f,pad in enumerate(self.finger_geoms):
                for k,box in enumerate(self.box_geoms):
                    if pair=={pad,box}:
                        wrench=np.zeros(6);mj.mj_contactForce(self.m,self.d,index,wrench)
                        forces[f,k]+=max(0.,float(wrench[0]))
        return forces

    def gripper(self,opening,seconds=.8):
        q=self.goal.copy();q[6:]=opening
        self.move(q,seconds)

    def grasp(self):
        """Close fingers and verify bilateral physical contact; creates no constraint."""
        self.gripper(.075)
        forces=self.grip_contacts()
        candidates=[k for k in range(2) if min(forces[:,k])>.3]
        self.grasp_events.append({'time_s':float(self.d.time),'normal_forces_n':forces.tolist(),'bilateral_candidates':candidates})
        if len(candidates)!=1:
            raise RuntimeError('grasp failed: both fingers must contact the same parcel')
        if self.r is not None:Image.fromarray(self.gripper_rgb()).save(self.out/f'grasp_{len(self.records)+1}.png')
        self.held=candidates[0]
        return self.held

    def execute(self,command):
        start_time=float(self.d.time); start_col=self.collisions; start_furniture=self.furniture_contacts
        rec={'command':command,'success':False}; self.command=command
        try:
            self.stage='PARSE'; intent=self.lang.predict(command); self.intent=intent; rec['intent']=intent; self.step(.8)
            self.stage='PERCEIVE'; rgb=self.rgb(); self.vision_frame=rgb
            det=self.vision.detect(rgb,intent['color']); self.detection=det; rec['detection']=det; self.step(.8)
            target=np.array(det['xy'])
            self.stage='APPROACH'; self.travel((-2.6,float(target[1])))
            self.stage='PICK'; self.gripper(.15); self.arm(target,.95); self.arm(target,.69,1.8)
            k=self.grasp(); rec['selected_parcel']=k
            self.arm(target,1.02,1.8)
            if self.d.xpos[self.parcels[k],2]<.82:raise RuntimeError('parcel did not lift with the fingers')
            self.monitor_grasp=True
            self.grasp_events.append({'phase':'LIFT','parcel_position':self.d.xpos[self.parcels[k]].tolist(),'forces':self.grip_contacts().tolist()})
            if self.r is not None:Image.fromarray(self.gripper_rgb()).save(self.out/f'lifted_{len(self.records)+1}.png')
            rec['lifted_height_m']=float(self.d.xpos[self.parcels[k],2])
            # Stow within inflated footprint before travelling.
            self.arm(self.base_xy()+[.25,0],1.02,1.8)
            self.grasp_events.append({'phase':'STOW','parcel_position':self.d.xpos[self.parcels[k]].tolist(),'forces':self.grip_contacts().tolist()})
            self.stage='TRANSPORT'; destination=DESTINATIONS[intent['destination']]
            route=self.travel(destination); rec['route']=route
            self.grasp_events.append({'phase':'ARRIVED','parcel_position':self.d.xpos[self.parcels[k]].tolist(),'forces':self.grip_contacts().tolist()})
            self.stage='PLACE'; drop=np.array([2.6,destination[1]])
            self.arm(drop,1.02); self.monitor_grasp=False; self.arm(drop,.69,1.8)
            self.grasp_events.append({'phase':'BEFORE_RELEASE','parcel_position':self.d.xpos[self.parcels[k]].tolist(),'forces':self.grip_contacts().tolist()})
            self.gripper(.15); self.held=None; self.step(.8)
            self.arm(drop,1.03)
            self.arm(self.base_xy()+[.25,0],1.03)
            pos=self.d.xpos[self.parcels[k]].copy()
            rec['final_parcel_position']=pos.tolist(); rec['placement_error_m']=float(np.linalg.norm(pos[:2]-drop))
            rec['obstacle_contact_steps']=self.collisions-start_col
            rec['robot_table_contact_steps']=self.furniture_contacts-start_furniture
            rec['success']=bool(rec['placement_error_m']<.15 and abs(pos[2]-.59)<.05 and rec['obstacle_contact_steps']==0 and rec['robot_table_contact_steps']==0)
            if not rec['success']: raise RuntimeError('placement or collision verification failed')
            self.stage='DONE'; self.step(1.2)
        except (ValueError,RuntimeError) as exc:
            rec['error']=str(exc); self.monitor_grasp=False;self.goal[:2]=0; self.stage='ERROR'; self.step(.5)
        rec['simulated_duration_s']=float(self.d.time-start_time)
        self.records.append(rec)
        self.save_results()
        return rec

    def save_results(self):
        (self.out/'results.json').write_text(json.dumps({'simulator':'MuJoCo '+mj.__version__,'simplifications':['differential wheel drive with floor contact; simulator pose feedback','SCARA arm with two contact-friction fingers, no weld','known map and fixed calibrated overhead camera','language backend: '+getattr(self.lang,'name','naive_bayes')],'vision_backend':getattr(getattr(self,'vision',None),'name','gaussian'),'vision_weights_sha256':getattr(getattr(self,'vision',None),'sha256',None),'language_backend':getattr(self.lang,'name','naive_bayes'),'language_model':getattr(self.lang,'model',None),'layout':self.layout_metadata,'distractors':self.distractor_metadata,'drive':{'type':'differential_wheels','wheel_distance_m':self.wheel_distance_m.tolist(),'wheel_contact_steps':self.wheel_contact_steps,'navigation':self.navigation,'robot_table_contact_steps':self.furniture_contacts},'gripper':{'type':'parallel_fingers','equality_constraints':int(self.m.neq),'events':self.grasp_events},'video_speed':self.video_speed,'records':self.records,'obstacle_contact_steps':self.collisions,'simulated_time_s':float(self.d.time)},ensure_ascii=False,indent=2))

    def frame(self):
        self.r.disable_segmentation_rendering(); self.r.update_scene(self.d,camera=self.cam)
        view=self.r.render().copy()
        canvas=Image.new('RGB',(960,640),(16,23,34)); canvas.paste(Image.fromarray(view),(0,85))
        draw=ImageDraw.Draw(canvas)
        try: font=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',18)
        except OSError: font=ImageFont.load_default(size=18)
        layout_label=(' | Seed '+str(self.layout_metadata['seed'])) if self.layout_metadata.get('seed') is not None else (' | Custom layout' if self.layout_metadata['mode']=='custom' else '')
        draw.text((24,16),'AI WAREHOUSE | PICK & DELIVER'+layout_label,fill='#6ae0d4',font=font)
        draw.text((24,47),self.command or 'Preparing warehouse',fill='white',font=font)
        draw.text((666,100),'STATE',fill='#8599b2',font=font)
        draw.text((666,132),self.stage,fill='#6ae0d4',font=font)
        stage_ja={'INITIALIZE':'準備','PARSE':'指示を解釈','PERCEIVE':'画像から箱を認識','APPROACH':'荷物へ接近','PICK':'指で挟んで持ち上げ','TRANSPORT':'障害物を避けて配送','PLACE':'指定場所に配置','DONE':'配送完了','ERROR':'安全停止'}
        draw.text((666,161),stage_ja.get(self.stage,''),fill='white',font=font)
        draw.text((666,193),f'Simulation: {self.d.time:5.1f}s',fill='white',font=font)
        draw.text((666,220),f'Obstacle contacts: {self.collisions}',fill='white',font=font)
        closeup=self.stage in ['PICK','PLACE']
        draw.text((666,247),'GRIPPER / physical contact' if closeup else ('YOLO26n / source-bay zoom' if getattr(getattr(self,'vision',None),'name','')=='yolo26n' else 'RGB CAMERA / learned model'),fill='#8599b2',font=font)
        if closeup:
            canvas.paste(Image.fromarray(self.gripper_rgb()).resize((280,210)),(666,290))
        elif self.vision_frame is not None:
            cam=Image.fromarray(self.vision_frame.copy())
            if self.detection:
                dr=ImageDraw.Draw(cam);x,y=self.detection['pixel_center']
                for box in self.detection.get('detections',[]):
                    color='#ff7777' if box['class']=='red_cube' else '#66aaff'
                    dr.rectangle(box['bbox_xyxy'],outline=color,width=2)
                    x1,y1=box['bbox_xyxy'][:2]
                    dr.text((x1,max(0,y1-13)),box['class']+f" {box['score']:.2f}",fill=color)
                dr.ellipse((x-3,y-3,x+3,y+3),outline='#00ff99',width=2)
            if getattr(getattr(self,'vision',None),'name','')=='yolo26n':
                zoom=ImageOps.contain(cam.crop((95,145,260,335)),(280,210))
                canvas.paste(zoom,(666+(280-zoom.width)//2,290))
            else:
                canvas.paste(cam.resize((280,210)),(666,290))
        if self.detection and 'score' in self.detection:
            x,y=self.detection['xy']
            draw.text((666,274),f"score {self.detection['score']:.2f} | x {x:.2f} y {y:.2f}m",fill='#c7d5e5',font=font.font_variant(size=12) if hasattr(font,'font_variant') else font)
        if self.intent:
            draw.text((666,520),self.intent['color']+' > '+self.intent['destination'],fill='#6ae0d4',font=font)
        if getattr(self.lang,'name','')=='local_llm':
            draw.text((666,548),'LLM: Qwen3 4B / local',fill='#9bacc0',font=font)
        draw.text((24,579),'YOLO26n on full RGB / display zoomed | wheel drive + contact fingers' if getattr(getattr(self,'vision',None),'name','')=='yolo26n' else 'Wheel-driven base + two-finger SCARA arm | known map / simulator pose',fill='#9bacc0',font=font)
        draw.text((24,608),f'Video: simulation time / {self.video_speed}倍速 ({self.video_speed}x playback, 15 fps)',fill='#9bacc0',font=font)
        if self.writer: self.writer.append_data(np.array(canvas))
        if self.frames%150==0: canvas.save(self.out/'preview.png')
        self.frames+=1
