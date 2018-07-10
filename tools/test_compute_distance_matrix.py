import numpy as np
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import compute_distance_matrix

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"
rawfile="data/text/STS.input.track5.en-en.txt"

vecs1 = wbc.get_vectors(wec_ids, {}, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0, skiprows=245)], raw=True)
vecs2 = wbc.get_vectors(wec_ids, {}, for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=1, skiprows=245)], raw=True)

print(compute_distance_matrix(vecs1, vecs2))
