"""Differential-drive kinematics and geometric path helpers."""
import math

WHEEL_RADIUS=.12
TRACK_WIDTH=.57

def wheel_speeds(v,w):
    return ((v-.5*TRACK_WIDTH*w)/WHEEL_RADIUS,
            (v+.5*TRACK_WIDTH*w)/WHEEL_RADIUS)

def wrap(angle):return math.atan2(math.sin(angle),math.cos(angle))

def corners(route):
    if len(route)<3:return list(route)
    result=[route[0]]
    prev=(route[1][0]-route[0][0],route[1][1]-route[0][1])
    for i in range(1,len(route)-1):
        nxt=(route[i+1][0]-route[i][0],route[i+1][1]-route[i][1])
        if abs(prev[0]*nxt[1]-prev[1]*nxt[0])>1e-8 or prev[0]*nxt[0]+prev[1]*nxt[1]<0:
            result.append(route[i])
        prev=nxt
    result.append(route[-1])
    return result
