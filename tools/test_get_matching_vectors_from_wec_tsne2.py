import sys
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_tsne

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

#wec_ids=sys.argv[1]
wec_ids="dataset:dblp;dims:200;fold:1;unit:stem;algo:cbow;iters:{10,20}"
pattern=sys.argv[1]

#pattern="*compu*"
highlight="" if len(sys.argv)<3 else sys.argv[2]    
#suppress = ".*_.*"
#suppress=""
#vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, exclude_pattern="*_*", label=pattern)

lim=(-10,10)
vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, exclude_pattern="", label=pattern)

plot_tsne(vecs, iters=1000, fontsize=14, size=(10,10), highlight=highlight, suppress=".*_.*", arrange_by=wec_ids, silent=True, share_axes=('none','none'), x_lim=lim, y_lim=lim, pdf_name="temp/cbow_random_wo_phrases3.pdf")

plot_tsne(vecs, iters=1000, fontsize=14, size=(10,10), highlight=highlight, suppress="", arrange_by=wec_ids, silent=True, share_axes=('none','none'), x_lim=lim, y_lim=lim, pdf_name="temp/cbow_random_w_phrases3.pdf")

