from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
importpath="data/embeddings/glove.6B/"

wbc = wb_conn(path=wbpath, create_if_missing=True)

for d in ['50', '100', '200', '300']:
    wbc.import_from_file(importpath+"glove.6B."+d+"d.txt", "algo:glove;dataset:6b;dims:"+d+";fold:1;unit:token")
    wbc.import_from_file(importpath+"glove.6B."+d+"d.txt", "algo:glove;dataset:6b;dims:"+d+";fold:1;unit:token;norm:l2", normalize="l2")