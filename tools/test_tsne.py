from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_tsne

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"

vecs = wbc.get_vectors(wec_ids, {}, for_input=[['this','is','a', 'test'], ['yet', 'another', 'test']])

plot_tsne(vecs)


