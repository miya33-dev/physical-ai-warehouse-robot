#!/usr/bin/env python3
"""Run with an installed environment, or the optional local .packages folder."""
from pathlib import Path
import sys
root=Path(__file__).resolve().parent
if (root/'.packages').exists(): sys.path.insert(0,str(root/'.packages'))
sys.path.insert(0,str(root/'src'))
# Source tree staging supports warehouse_robot at the root as well.
import argparse
p=argparse.ArgumentParser()
p.add_argument('--command',default='赤い箱を出荷場所へ運んで')
p.add_argument('--video',action='store_true')
p.add_argument('--demo',action='store_true',help='Run red/shipping then blue/storage')
p.add_argument('--output',default=str(root/'outputs'/'first_run'))
a=p.parse_args()
from warehouse_robot.simulation import Simulation
s=Simulation(a.output,video=a.video)
try:
    print('Training pixel classifier:',s.train_vision(),flush=True)
    commands=['赤い箱を出荷場所へ運んで','青い箱を保管場所へ運んで'] if a.demo else [a.command]
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
