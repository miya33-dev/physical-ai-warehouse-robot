"""A* on a known occupancy map; this module is not a learned AI."""
import heapq
import math

RES = .10
RADIUS = .53
OBSTACLE = (-.30, .30, -.68, .68)

def blocked(x, y):
    a,b,c,d=OBSTACLE
    return a-RADIUS <= x <= b+RADIUS and c-RADIUS <= y <= d+RADIUS

def plan(start, goal, trace=None):
    def cell(p): return tuple(round(v/RES) for v in p)
    s,g=cell(start),cell(goal)
    def valid(n):
        x,y=n[0]*RES,n[1]*RES
        return -2.7<=x<=2.0 and -2.1<=y<=2.1 and not blocked(x,y)
    if not valid(s) or not valid(g): raise ValueError('start or goal is blocked')
    if trace is not None:
        trace.update(start=list(start),goal=list(goal),resolution=RES,radius=RADIUS,obstacle=list(OBSTACLE),bounds=[-2.7,2.,-2.1,2.1],events=[],path=[])
    q=[(0,s)]; costs={s:0}; parents={}
    while q:
        _,n=heapq.heappop(q)
        event={'current':list(n),'g':costs[n],'h':abs(n[0]-g[0])+abs(n[1]-g[1]),'opened':[]}
        if trace is not None:trace['events'].append(event)
        if n==g:
            path=[n]
            while n!=s: n=parents[n]; path.append(n)
            route=[(x*RES,y*RES) for x,y in reversed(path)]
            if trace is not None:trace['path']=[list(p) for p in route]
            return route
        for dx,dy in [(1,0),(-1,0),(0,1),(0,-1)]:
            nn=(n[0]+dx,n[1]+dy)
            if not valid(nn): continue
            cost=costs[n]+1
            if cost < costs.get(nn,math.inf):
                costs[nn]=cost; parents[nn]=n
                if trace is not None:event['opened'].append(list(nn))
                heapq.heappush(q,(cost+abs(nn[0]-g[0])+abs(nn[1]-g[1]),nn))
    raise ValueError('no safe route')
