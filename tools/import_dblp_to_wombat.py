from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
importpath="data/embeddings/"

wbc = wb_conn(path=wbpath, create_if_missing=True)

for it in ['5', '10', '20', '50']:
    for al in ['cbow', 'sg']:
        wbc.import_from_file(importpath+"cso-phrases."+al+".200d.mc5.w5.iters"+it+".vec", "algo:"+al+";dataset:dblp;dims:200;fold:1;unit:stem;iters:"+it)
