import numpy as np
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import compute_distance_matrix, plot_heatmap

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:{50,100};fold:1;unit:token"
rawfile="data/text/STS.input.track5.en-en.txt"

vecs1 = wbc.get_vectors(wec_ids, {}, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0, skiprows=0)], raw=True, in_order=True)
vecs2 = wbc.get_vectors(wec_ids, {}, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=1, skiprows=0)], raw=True, in_order=True)

distance_matrices = compute_distance_matrix(vecs1, vecs2, invert=True, ignore_matching=False)
for mi, r in enumerate(distance_matrices):
    for ti, (matrix, xwords, ywords, xstring, ystring) in enumerate(r):
        plot_heatmap(matrix, xwords, ywords, xstring=xstring, ystring=ystring, plot_name="temp/"+str(mi)+"_"+str(ti)+".png")
