import sys
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_tsne

for i in range(len(sys.argv)):
    if sys.argv[i]=="-wbpath":
        wbpath=sys.argv[i+1]
    elif sys.argv[i]=="-wecs":
        wec_ids=sys.argv[i+1]
    elif sys.argv[i]=="-seed":
        seed=int(sys.argv[i+1])
    elif sys.argv[i]=="-size":
        size=sys.argv[i+1]
    
        
wbc = wb_conn(path=wbpath, create_if_missing=False)
vecs = wbc.sample_words(wec_ids, seed=seed, size=size)
print(vecs)

