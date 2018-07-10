import sys
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_tsne

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids=sys.argv[1]
pattern=sys.argv[2]

highlight="" if len(sys.argv)<4 else sys.argv[3]

vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, label=pattern)
plot_tsne(vecs, iters=1000, fontsize=5, size=(10,10), highlight=highlight, arrange_by=wec_ids, silent=False)

