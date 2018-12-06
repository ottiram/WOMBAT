import sys
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import get_most_similar
import scipy.spatial.distance as dist

wbpath=sys.argv[1]
wec_ids=sys.argv[2]
targets=sys.argv[3].split(",")
try:
    to_rank=sys.argv[4].split(",")
except IndexError:
    to_rank=[]

wbc = wb_conn(path=wbpath, create_if_missing=False)

sims = get_most_similar(wbc, wec_ids, targets=targets, measures=[dist.cosine], to_rank=to_rank)
for (w, wec, mes, simlist) in sims:
    print("\n%s"%(wec))
    for (t,s) in simlist:
        print("%s(%s, %s)\t%s"%(mes,w,t,s))
    
