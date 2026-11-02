#!/usr/bin/env python3
"""Run with an installed environment, or the optional local .packages folder."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
if (root/'.packages').exists(): sys.path.insert(0,str(root/'.packages'))
if (root/'.vision-packages').exists():sys.path.insert(0,str(root/'.vision-packages'))
sys.path.insert(0,str(root/'src'))
# Source tree staging supports warehouse_robot at the root as well.
import argparse
p=argparse.ArgumentParser()
p.add_argument('--command',default='赤い箱を出荷場所へ運んで')
p.add_argument('--video',action='store_true')
p.add_argument('--explain-video',action='store_true',help='Show real LLM JSON and overhead A* search/trajectory')
p.add_argument('--video-speed',type=int,choices=[1,2],default=1)
p.add_argument('--vision',choices=['yolo','gaussian'],default='yolo')
p.add_argument('--vision-device',choices=['mps','cpu'],default='mps')
p.add_argument('--weights',type=Path,default=root/'models/cubes_yolo26n.pt')
p.add_argument('--language',choices=['llm','naive_bayes'],default='llm')
p.add_argument('--parse-only',action='store_true',help='Interpret the command without starting simulation')
layout_options=p.add_mutually_exclusive_group()
layout_options.add_argument('--layout-seed',type=int,help='Randomize source-table XY positions with a reproducible seed')
layout_options.add_argument('--layout-file',type=Path,help='JSON with red and blue XY positions')
p.add_argument('--distractors',action='store_true',help='Place static same-color distractors on unused table lanes')
p.add_argument('--demo',action='store_true',help='Run red/shipping then blue/storage')
p.add_argument('--output',default=str(root/'outputs'/'first_run'))
a=p.parse_args()
if a.language=='llm':
    from warehouse_robot.llm_language import LocalLLM
    language=LocalLLM()
else:
    from warehouse_robot.language import IntentModel
    language=IntentModel()
if a.parse_only:
    import json
    try:
        print(json.dumps(language.predict(a.command),ensure_ascii=False,indent=2))
    except ValueError as exc:
        print(str(exc),file=sys.stderr)
        sys.exit(2)
    sys.exit(0)
from warehouse_robot.simulation import Simulation
s=Simulation(a.output,video=False,language_model=language)
try:
    if a.vision=='gaussian':
        print('Training pixel classifier:',s.train_vision(),flush=True)
    else:
        from warehouse_robot.detector import CubeDetector
        s.vision=CubeDetector(a.weights,device=a.vision_device)
        s.vision_frame=s.rgb()
        print('Vision: fine-tuned YOLO26n, device='+a.vision_device,flush=True)
    if a.layout_seed is not None:
        from warehouse_robot.layout import random_layout
        s.set_layout(random_layout(a.layout_seed),seed=a.layout_seed)
    elif a.layout_file is not None:
        import json
        s.set_layout(json.loads(a.layout_file.read_text()))
    if a.distractors:s.enable_distractors()
    if a.video: s.start_video(speed=a.video_speed,explain=a.explain_video)
    demo_commands=['赤い荷物を発送エリアまで持っていって','青の荷物を倉庫にしまって'] if a.language=='llm' else ['赤い箱を出荷場所へ運んで','青い箱を保管場所へ運んで']
    commands=demo_commands if a.demo else [a.command]
    if a.demo: s.step(3.)
    results=[]
    for command in commands:
        result=s.execute(command); results.append(result)
        print({k:v for k,v in result.items() if k!='route'},flush=True)
        if not result['success']: break
    if a.demo: s.step(3.)
    s.frame()
    s.save_results()
    if not all(r['success'] for r in results): sys.exit(1)
finally: s.close()
