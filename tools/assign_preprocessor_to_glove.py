from wombat_api.preprocessors.standard_preprocessor import preprocessor
from wombat_api.core import connector as wb_conn

prepro=preprocessor(name="wombat_standard_preprocessor", phrasefile="")
prepro.pickle("temp/wombat_standard_preprocessor.pkl")

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)
wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token;norm:{none,abtt}", "temp/wombat_standard_preprocessor.pkl")

# Calling this method with an empty string as pickle file name removes the preprocessor.
# wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token;norm:{none,abtt}", "")
