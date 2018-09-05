import numpy as np, scipy.spatial.distance
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_pairwise_distances

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

# Note: You can use e.g. algo:glove;dataset:6b;dims:{50,100,200};fold:1;unit:token" 
# to create three different plots in one run!
wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"
rawfile="data/text/STS.input.track5.en-en.txt"

pp_cache={}
vecs1 = wbc.get_vectors(wec_ids, pp_cache, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0, skiprows=0)], raw=True)
vecs2 = wbc.get_vectors(wec_ids, pp_cache, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=1, skiprows=0)], raw=True)

pd = plot_pairwise_distances(vecs1, vecs2, arrange_by=wec_ids, pdf_name="temp/sent_sim.pdf", size=(25,10), max_pairs=20, ignore_identical=True)
