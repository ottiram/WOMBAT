from wombat_api.preprocessors import standard_preprocessor
from wombat_api.core import connector as wb_conn

prepro=standard_preprocessor.preprocessor(name="my_standard_preprocessor", phrasefile="")
prepro.pickle("temp/my_standard_preprocessor.pkl")

wbpath="data/wombat-data/"

wbc = wb_conn(path=wbpath, create_if_missing=True)
wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token", "temp/my_standard_preprocessor.pkl")
#wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token", "")

