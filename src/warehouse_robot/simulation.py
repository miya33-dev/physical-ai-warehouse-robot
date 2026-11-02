"""MuJoCo SCARA mobile manipulator, position-served XY base and suction weld.
Known map/calibration, no wheel dynamics or contact-finger grasping.
"""
from pathlib import Path
import json
import math
import numpy as np
import mujoco as mj
from PIL import Image,ImageDraw,ImageFont
from .planning import plan
from .language import IntentModel
from .vision import PixelModel

ROOT=Path(__file__).resolve().parents[1]
if ROOT.name=='src': ROOT=ROOT.parent
DESTINATIONS={'shipping':(1.9,-1.25),'storage':(1.9,1.25)}

class Simulation:
    def __init__(self,out,render=True,video=False):
        self.out=Path(out); self.out.mkdir(parents=True,exist_ok=True)
        self.m=mj.MjModel.from_xml_path(str(ROOT/'assets/warehouse.xml'))
        self.d=mj.MjData(self.m)
        self.r=mj.Renderer(self.m,height=480,width=640) if render else None
        self.cam=mj.MjvCamera(); self.cam.lookat[:]=[0,0,.3]; self.cam.distance=8.8; self.cam.azimuth=125; self.cam.elevation=-43
        self.joints=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_JOINT,n) for n in ['base_x','base_y','shoulder','elbow','lift']]
        self.qids=[int(self.m.jnt_qposadr[j]) for j in self.joints]
        self.goal=np.array([0,0,-.9,2.2,0.])
        self.d.qpos[self.qids]=self.goal; self.d.ctrl[:]=self.goal
        self.tip=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_SITE,'tip')
        self.tool=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'tool')
        self.obstacle=mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'obstacle')
        self.robot_geoms=set(range(mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,'chassis'),self.m.ngeom))
        self.parcels=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,n) for n in ['parcel_a','parcel_b']]
        self.box_geoms=[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_GEOM,n) for n in ['box_a','box_b']]
        self.contact_pairs=set(); self.collisions=0; self.held=None; self.stage='INITIALIZE'; self.command=''; self.intent=None
        self.frames=0; self.records=[]; self.trail=[]; self.vision_frame=None; self.detection=None
        self.lang=IntentModel(); self.writer=None
        if video:
            import imageio.v2 as iio
            self.writer=iio.get_writer(str(self.out/'demo.mp4'),fps=15,codec='libx264',quality=8,macro_block_size=16)
        mj.mj_forward(self.m,self.d)
        self.step(.6)

    def close(self):
        if self.writer: self.writer.close(); self.writer=None
        if self.r: self.r.close(); self.r=None

    def base_xy(self): return self.d.xpos[mj.mj_name2id(self.m,mj.mjtObj.mjOBJ_BODY,'base'),:2].copy()

    def step(self,seconds):
        for i in range(round(seconds/self.m.opt.timestep)):
            self.d.ctrl[:]=self.goal
            mj.mj_step(self.m,self.d)
            for c in self.d.contact:
                a,b=int(c.geom1),int(c.geom2)
                if self.obstacle in (a,b) and (a in self.robot_geoms or b in self.robot_geoms or a in self.box_geoms or b in self.box_geoms):
                    if c.dist<-.001:
                        self.collisions+=1; self.contact_pairs.add(tuple(sorted((a,b))))
            if self.writer and round(self.d.time/self.m.opt.timestep)%17==0:
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
        x,y=np.array(xy)-self.base_xy()
        cc=(x*x+y*y-2*.45**2)/(2*.45**2)
        if not -1<=cc<=1: raise ValueError('target outside arm reach')
        e=math.acos(cc)
        a=math.atan2(y,x)-math.atan2(.45*math.sin(e),.45+.45*math.cos(e))
        q=self.goal.copy(); q[2:]=[a,e,z-1.135]
        self.move(q,seconds)
        err=float(np.linalg.norm(self.d.site_xpos[self.tip]-np.array([xy[0],xy[1],z])))
        if err>.04: raise RuntimeError(f'arm tracking error {err:.3f} m')

    def travel(self,xy):
        route=plan(self.base_xy(),xy)
        self.trail.extend(route)
        for x,y in route[1:]:
            q=self.goal.copy(); q[:2]=[x+2.6,y]
            # Each grid segment moves at <= 0.30 m/s. No teleports.
            self.move(q,.36)
        if np.linalg.norm(self.base_xy()-xy)>.08: raise RuntimeError('base failed to reach goal')
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
        (self.out/'training_metrics.json').write_text(json.dumps({'seed':42,'training_samples':self.vision.n_train,'language_training_samples':self.lang.n_train,'nominal_heldout_pixel_accuracy':acc,'note':'synthetic fixed-camera scene only; not real-world validation'},indent=2))
        Image.fromarray(rgb).save(self.out/'camera.png')
        return acc

    def attach(self):
        tip=self.d.site_xpos[self.tip].copy()
        # Physical proximity validation selects a body only after RGB targeting.
        distances=[np.linalg.norm(self.d.xpos[b]-tip) for b in self.parcels]
        k=int(np.argmin(distances))
        if distances[k]>.14: raise RuntimeError('suction tool too far from parcel')
        b=self.parcels[k]
        rot=self.d.xmat[self.tool].reshape(3,3)
        rel=rot.T@(self.d.xpos[b]-self.d.xpos[self.tool])
        # MuJoCo weld data: anchor[0:3], relative position[3:6], quaternion[6:10].
        self.m.eq_data[k,3:6]=rel
        qa=np.empty(4); qb=np.empty(4); inv=np.empty(4); relq=np.empty(4)
        mj.mju_mat2Quat(qa,self.d.xmat[self.tool]); mj.mju_mat2Quat(qb,self.d.xmat[b])
        mj.mju_negQuat(inv,qa); mj.mju_mulQuat(relq,inv,qb)
        self.m.eq_data[k,6:10]=relq
        self.d.eq_active[k]=1; self.held=k
        self.step(.3)
        return k

    def execute(self,command):
        start_time=float(self.d.time); start_col=self.collisions
        rec={'command':command,'success':False}; self.command=command
        try:
            self.stage='PARSE'; intent=self.lang.predict(command); self.intent=intent; rec['intent']=intent; self.step(.8)
            self.stage='PERCEIVE'; rgb=self.rgb(); self.vision_frame=rgb
            det=self.vision.detect(rgb,intent['color']); self.detection=det; rec['detection']=det; self.step(.8)
            target=np.array(det['xy'])
            self.stage='APPROACH'; self.travel((-2.6,float(target[1])))
            self.stage='PICK'; self.arm(target,.95); self.arm(target,.69,1.8)
            k=self.attach(); rec['selected_parcel']=k
            self.arm(target,1.02,1.8)
            # Stow within inflated footprint before travelling.
            self.arm(self.base_xy()+[.25,0],1.02,1.8)
            self.stage='TRANSPORT'; destination=DESTINATIONS[intent['destination']]
            route=self.travel(destination); rec['route']=route
            self.stage='PLACE'; drop=np.array([2.6,destination[1]])
            self.arm(drop,1.02); self.arm(drop,.69,1.8)
            self.d.eq_active[k]=0; self.held=None; self.step(.8)
            self.arm(drop,1.03)
            self.arm(self.base_xy()+[.25,0],1.03)
            pos=self.d.xpos[self.parcels[k]].copy()
            rec['final_parcel_position']=pos.tolist(); rec['placement_error_m']=float(np.linalg.norm(pos[:2]-drop))
            rec['obstacle_contact_steps']=self.collisions-start_col
            rec['success']=bool(rec['placement_error_m']<.15 and abs(pos[2]-.59)<.05 and rec['obstacle_contact_steps']==0)
            if not rec['success']: raise RuntimeError('placement or collision verification failed')
            self.stage='DONE'; self.step(1.2)
        except (ValueError,RuntimeError) as exc:
            rec['error']=str(exc); self.stage='ERROR'; self.step(.5)
        rec['simulated_duration_s']=float(self.d.time-start_time)
        self.records.append(rec)
        self.save_results()
        return rec

    def save_results(self):
        (self.out/'results.json').write_text(json.dumps({'simulator':'MuJoCo '+mj.__version__,'simplifications':['XY position-served base, no wheel dynamics','SCARA arm with suction weld','known map and fixed calibrated overhead camera','small trained classifiers; not an LLM'],'records':self.records,'obstacle_contact_steps':self.collisions,'simulated_time_s':float(self.d.time)},ensure_ascii=False,indent=2))

    def frame(self):
        self.r.disable_segmentation_rendering(); self.r.update_scene(self.d,camera=self.cam)
        view=self.r.render().copy()
        canvas=Image.new('RGB',(960,640),(16,23,34)); canvas.paste(Image.fromarray(view),(0,85))
        draw=ImageDraw.Draw(canvas)
        try: font=ImageFont.truetype('/System/Library/Fonts/ヒラギノ角ゴシック W3.ttc',18)
        except OSError: font=ImageFont.load_default(size=18)
        draw.text((24,16),'AI WAREHOUSE | PICK & DELIVER',fill='#6ae0d4',font=font)
        draw.text((24,47),self.command or 'Preparing warehouse',fill='white',font=font)
        draw.text((666,100),'STATE',fill='#8599b2',font=font)
        draw.text((666,132),self.stage,fill='#6ae0d4',font=font)
        stage_ja={'INITIALIZE':'準備','PARSE':'指示を解釈','PERCEIVE':'画像から箱を認識','APPROACH':'荷物へ接近','PICK':'吸着して持ち上げ','TRANSPORT':'障害物を避けて配送','PLACE':'指定場所に配置','DONE':'配送完了','ERROR':'安全停止'}
        draw.text((666,161),stage_ja.get(self.stage,''),fill='white',font=font)
        draw.text((666,193),f'Simulation: {self.d.time:5.1f}s',fill='white',font=font)
        draw.text((666,220),f'Obstacle contacts: {self.collisions}',fill='white',font=font)
        draw.text((666,247),'RGB CAMERA / learned model',fill='#8599b2',font=font)
        if self.vision_frame is not None:
            cam=Image.fromarray(self.vision_frame.copy())
            if self.detection:
                dr=ImageDraw.Draw(cam); x,y=self.detection['pixel_center']; dr.ellipse((x-15,y-15,x+15,y+15),outline='#00ff99',width=3)
            canvas.paste(cam.resize((280,210)),(666,290))
        if self.intent:
            draw.text((666,520),self.intent['color']+' > '+self.intent['destination'],fill='#6ae0d4',font=font)
        draw.text((24,579),'Prototype: XY servo base + SCARA suction arm | fixed map',fill='#9bacc0',font=font)
        draw.text((24,608),'Video timing: simulation time (15 fps; ~1x)',fill='#9bacc0',font=font)
        if self.writer: self.writer.append_data(np.array(canvas))
        if self.frames%150==0: canvas.save(self.out/'preview.png')
        self.frames+=1
