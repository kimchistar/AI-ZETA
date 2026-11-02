import glob, os, numpy as np, tifffile
from config import LAUE_DIRS as CASES
_cache={}
def files(case):
    if case not in _cache:
        base=CASES[case]
        _cache[case]=sorted(glob.glob(os.path.join(base,"*.tif*")))
    return _cache[case]
def load(case,i):
    return tifffile.imread(files(case)[i]).astype(np.float32)
def disp(a,lo=1,hi=99.7):
    l,h=np.percentile(a,[lo,hi]); return np.clip((a-l)/(h-l+1e-9),0,1)
