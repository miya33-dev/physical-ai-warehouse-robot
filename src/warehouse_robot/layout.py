"""Source-table layouts. Placement settings are not perception input."""
import math
import random

X_BOUNDS=(-2.10,-1.80)
Y_BOUNDS=(-.65,.65)
MIN_SEPARATION=.28

def validate_layout(layout):
    if not isinstance(layout,dict) or set(layout)!={'red','blue'}:
        raise ValueError('layout must contain exactly red and blue XY positions')
    result={}
    for color in ['red','blue']:
        xy=layout[color]
        if not isinstance(xy,(list,tuple)) or len(xy)!=2:
            raise ValueError(color+' position must be [x, y]')
        if any(isinstance(v,bool) or not isinstance(v,(float,int)) or not math.isfinite(v) for v in xy):
            raise ValueError('positions must be finite numbers')
        x,y=map(float,xy)
        if not (X_BOUNDS[0]<=x<=X_BOUNDS[1] and Y_BOUNDS[0]<=y<=Y_BOUNDS[1]):
            raise ValueError('parcel must be within the supported source-table bounds')
        result[color]=[x,y]
    # Keep distinct pickup lanes and space for the suction head.
    if abs(result['red'][1]-result['blue'][1])<MIN_SEPARATION:
        raise ValueError('red and blue Y positions must be at least 0.28 m apart')
    return result

def random_layout(seed):
    if isinstance(seed,bool) or not isinstance(seed,int) or seed<0:
        raise ValueError('seed must be a nonnegative integer')
    rng=random.Random(seed)
    for _ in range(1000):
        layout={c:[rng.uniform(*X_BOUNDS),rng.uniform(*Y_BOUNDS)] for c in ['red','blue']}
        try:return validate_layout(layout)
        except ValueError:pass
    raise RuntimeError('could not sample a separated layout')
