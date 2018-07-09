# WOMBAT
Word Embedding Database

See <a href="http://arxiv.org/abs/1807.00717" target="_blank">this paper</a> (to appear at <a href="http://coling2018.org/accepted-demo-papers/" target="_blank">COLING 2018</a>) for more details.


<pre>
 |                                   | 
 |             ,.--""""--.._         |
 |           ."     .'      `-.      |
 |          ;      ;           ;     |
 |         '      ;             )    |
 |        /     '             . ;    |
 |       /     ;     `.        `;    |
 |     ,.'     :         .    : )    |
 |     ;|\'    :     `./|) \  ;/     |
 |     ;| \"  -,-  "-./ |;  ).;      |
 |     /\/             \/   );       |
 |    :                \     ;       |
 |    :     _      _     ;   )       |
 |    `.   \;\    /;/    ;  /        |
 |      !    :   :     ,/  ;         |
 |       (`. : _ : ,/     ;          |
 |        \\\`"^" ` :    ;           |   This is WOMBAT, the WOrd eMBedding dATa base API (Version 2.0)
 |                (    )             |
 |                 ////              |
 |                                   |
 | Wombat artwork by akg             |
 |            http://wombat.ascii.uk |
</pre>


<h3>Introduction</h3>
<p>
<b>WOMBAT</b>, the WOrd eMBedding dATabase, is a light-weight Python tool for more transparent, efficient, and robust access to potentially large numbers of word embedding collections (WECs). 
It supports NLP researchers and practitioners in developing compact, efficient, and reusable code. 
Key features of WOMBAT are
<ol type=1>
<li>transparent identification of WECs by means of a clean syntax and human-readable features, </li>
<li>efficient lazy, on-demand retrieval of word vectors, and</li> 
<li>increased robustness by systematic integration of executable preprocessing code. </li>
</ol>
</p>

<p>
WOMBAT implements some Best Practices for research reproducibility and complements existing approaches towards WEC standardization and sharing. 

WOMBAT provides a single point of access to existing WECs. Each plain text WEC file has to be imported into WOMBAT once, receiving in the process a set of ```ATT:VAL``` identifiers consisting of five system attributes (algo, dims, dataset, unit, fold) plus arbitrarily many user-defined ones.

</p>

<h3>Installation</h3>
<p>
WOMBAT does not have a lot of special requirements. The basic functionality only requires sqlite3, numpy, and tqdm. Note that sqlite3 is commonly available as a default package, e.g. with conda.
</p>

<p>
In addition, the standard_preprocessor (see below) requires NLTK 3.2.5. 
A minimal working environment can be set up like this:
</p>

<p>

```shell
$ conda create --name wombat python=3.6 numpy tqdm nltk==3.2.5
$ source activate wombat
$ git clone https://github.com/nlpAThits/WOMBAT.git
$ cd WOMBAT
$ pip install -e .
```
</p>

<p>
Note: Depending on your environment, you might have to install NLTK 3.2.5 with

```shell
conda install -c conda-forge nltk==3.2.5
```

</p>




<h3>Importing Pre-Trained Embeddings to WOMBAT: GloVe</h3>
<p>
One of the main uses of WOMBAT is as a wrapper for accessing existing, off-the-shelf word embeddings like e.g. GloVe. (The other use involves access to self-trained embeddings, including preprocessing and handling of multi-word-expressions, cf. below.)

The following code is sufficient to import a sub set of the GloVe embeddings. 
```python
from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
importpath="data/embeddings/"
wbc = wb_conn(path=wbpath, create_if_missing=True)
for d in ['50', '100', '200', '300']:
    wbc.import_from_file(importpath+"glove.6B."+d+"d.txt", 
                         "algo:glove;dataset:6b;dims:"+d+";fold:1;unit:token")
```
</p>
<p>
To execute this code, run 

```shell
$ python tools/import_to_wombat.py
```

from the WOMBAT directory.
</p>

<p>

The required GloVe embeddings are <b>not part of WOMBAT</b> and can be obtained from Stanford <a href="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip">here</a>. 
Extract them into ```data/embeddings ```. The WOMBAT master and embeddings data bases will be created in ```data/wombat-data```. 

</p>

<p>

The above import assigns only the minimally required ```ATT:VAL``` pairs to the embeddings.

<table>
<tr><td><b>Attribute</b></td><td><b>Meaning</b></td></tr>
<tr><td>algo</td><td>Descriptive label for the <b>algorithm</b> used for training these embeddding vectors.</td></tr>
<tr><td>dataset</td><td>Descriptive label for the <b>data set</b> used for training these embedding vectors.</td></tr>
<tr><td>dims</td><td><b>Dimensionality</b> of these embedding vectors. Required for description and for creating right-sized <b>empty</b> vectors for OOV words.</td></tr>
<tr><td>fold</td><td>Indicates whether the embedding vectors are <b>case-sensitive</b> (fold=0) or not (fold=1). If fold=1, input words are lowercased before lookup.</td></tr>
<tr><td>unit</td><td>Unit of representation used in the embedding vectors. Works as a descriptive label with pre-trained embeddings for which no custom preprocessing has been integrated into WOMBAT. If custom preprocessing exists, the value of this attribute is passed to the process() method. The current preprocessor modules support the values <b>stem</b> and <b>token</b>.</td></tr>
</table>
</p>

<p>
After import, the embedding vectors are immediately available for efficient lookup of <b>already preprocessed</b> words. 
The following code accesses one of the four GloVe WECs and looks up &lt;unit, vector&gt; tuples for two sequences of words.
</p>

```python
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"

vecs = wbc.get_vectors(wec_ids, {}, for_input=[['this','is','a', 'test'], ['yet', 'another', 'test']])

# One wec_result for each wec specified in wec_ids
for wec_index in range(len(vecs)):
    # Index 0 element is the wec_id
    print("\nWEC: %s"%vecs[wec_index][0])
    # Index 1 element is the list of all results for this wec
    # Result list contains tuples of ([raw],[prepro],[(w,v) tuples])
    for (raw, prepro, tuples) in vecs[wec_index][1]: 
        print("Raw:    '%s'"%str(raw))
        print("Prepro: %s"%str(prepro))
        for (w,v) in tuples:
            print("Unit:   %s\nVector: %s\n"%(w,str(v)))
```

<p>
To execute this code, run 

```shell
$ python tools/test_get_vectors.py
```

from the WOMBAT directory.
</p>

<p>
The result is a nested python list.
</p>

<pre>
WEC: algo:glove;dataset:6b;dims:50;fold:1;unit:token
Raw:    ''
Prepro: ['this', 'is', 'a', 'test']
Unit:   a
Vector: [ 0.21705   0.46515  -0.46757   0.10082   1.0135    0.74845  -0.53104
 -0.26256   0.16812   0.13182  -0.24909  -0.44185  -0.21739   0.51004
  0.13448  -0.43141  -0.03123   0.20674  -0.78138  -0.20148  -0.097401
  0.16088  -0.61836  -0.18504  -0.12461  -2.2526   -0.22321   0.5043
  0.32257   0.15313   3.9636   -0.71365  -0.67012   0.28388   0.21738
  0.14433   0.25926   0.23434   0.4274   -0.44451   0.13813   0.36973
 -0.64289   0.024142 -0.039315 -0.26037   0.12017  -0.043782  0.41013
  0.1796  ]

Unit:   is
Vector: [ 6.1850e-01  6.4254e-01 -4.6552e-01  3.7570e-01  7.4838e-01  5.3739e-01
  2.2239e-03 -6.0577e-01  2.6408e-01  1.1703e-01  4.3722e-01  2.0092e-01
 -5.7859e-02 -3.4589e-01  2.1664e-01  5.8573e-01  5.3919e-01  6.9490e-01
 -1.5618e-01  5.5830e-02 -6.0515e-01 -2.8997e-01 -2.5594e-02  5.5593e-01
  2.5356e-01 -1.9612e+00 -5.1381e-01  6.9096e-01  6.6246e-02 -5.4224e-02
  3.7871e+00 -7.7403e-01 -1.2689e-01 -5.1465e-01  6.6705e-02 -3.2933e-01
  1.3483e-01  1.9049e-01  1.3812e-01 -2.1503e-01 -1.6573e-02  3.1200e-01
 -3.3189e-01 -2.6001e-02 -3.8203e-01  1.9403e-01 -1.2466e-01 -2.7557e-01
  3.0899e-01  4.8497e-01]

Unit:   test
Vector: [ 0.13175  -0.25517  -0.067915  0.26193  -0.26155   0.23569   0.13077
 -0.011801  1.7659    0.20781   0.26198  -0.16428  -0.84642   0.020094
  0.070176  0.39778   0.15278  -0.20213  -1.6184   -0.54327  -0.17856
  0.53894   0.49868  -0.10171   0.66265  -1.7051    0.057193 -0.32405
 -0.66835   0.26654   2.842     0.26844  -0.59537  -0.5004    1.5199
  0.039641  1.6659    0.99758  -0.5597   -0.70493  -0.0309   -0.28302
 -0.13564   0.6429    0.41491   1.2362    0.76587   0.97798   0.58507
 -0.30176 ]

Unit:   this
Vector: [ 5.3074e-01  4.0117e-01 -4.0785e-01  1.5444e-01  4.7782e-01  2.0754e-01
 -2.6951e-01 -3.4023e-01 -1.0879e-01  1.0563e-01 -1.0289e-01  1.0849e-01
 -4.9681e-01 -2.5128e-01  8.4025e-01  3.8949e-01  3.2284e-01 -2.2797e-01
 -4.4342e-01 -3.1649e-01 -1.2406e-01 -2.8170e-01  1.9467e-01  5.5513e-02
  5.6705e-01 -1.7419e+00 -9.1145e-01  2.7036e-01  4.1927e-01  2.0279e-02
  4.0405e+00 -2.4943e-01 -2.0416e-01 -6.2762e-01 -5.4783e-02 -2.6883e-01
  1.8444e-01  1.8204e-01 -2.3536e-01 -1.6155e-01 -2.7655e-01  3.5506e-02
 -3.8211e-01 -7.5134e-04 -2.4822e-01  2.8164e-01  1.2819e-01  2.8762e-01
  1.4440e-01  2.3611e-01]

Raw:    ''
Prepro: ['yet', 'another', 'test']
Unit:   another
Vector: [ 0.50759    0.26321    0.19638    0.18407    0.90792    0.45267
 -0.54491    0.41816    0.039569   0.061854  -0.24574   -0.38502
 -0.39649    0.32165    0.59611   -0.3997    -0.015734   0.074218
 -0.83148   -0.019284  -0.21331    0.12873   -0.2541     0.079348
  0.12588   -2.1294    -0.29092    0.044597   0.27354   -0.037492
  3.458     -0.34642   -0.32803    0.17566    0.22467    0.08987
  0.24528    0.070129   0.2165    -0.44313    0.02516    0.40817
 -0.33533    0.0067758  0.11499   -0.15701   -0.085219   0.018568
  0.26125    0.015387 ]

Unit:   test
Vector: [ 0.13175  -0.25517  -0.067915  0.26193  -0.26155   0.23569   0.13077
 -0.011801  1.7659    0.20781   0.26198  -0.16428  -0.84642   0.020094
  0.070176  0.39778   0.15278  -0.20213  -1.6184   -0.54327  -0.17856
  0.53894   0.49868  -0.10171   0.66265  -1.7051    0.057193 -0.32405
 -0.66835   0.26654   2.842     0.26844  -0.59537  -0.5004    1.5199
  0.039641  1.6659    0.99758  -0.5597   -0.70493  -0.0309   -0.28302
 -0.13564   0.6429    0.41491   1.2362    0.76587   0.97798   0.58507
 -0.30176 ]

Unit:   yet
Vector: [ 0.6935   -0.13892  -0.10862  -0.18671   0.56311   0.070388 -0.52788
  0.35681  -0.21765   0.44888  -0.14023   0.020312 -0.44203   0.072964
  0.85846   0.41819   0.19097  -0.33512   0.012309 -0.53561  -0.44548
  0.38117   0.2255   -0.26948   0.56835  -1.717    -0.7606    0.43306
  0.4189    0.091699  3.2262   -0.18561  -0.014535 -0.69816   0.21151
 -0.28682   0.12492   0.49278  -0.57784  -0.75677  -0.47876  -0.083749
 -0.013377  0.19862  -0.14819   0.21787  -0.30472   0.54255  -0.20916
  0.14965 ]
</pre>


<p>
WOMBAT also supports the selection of embedding vectors for words <b>matching a particular string pattern</b>.
The following code looks up embedding vectors matching the supplied pattern. The pattern uses the GLOB syntax described <a href=http://www.sqlitetutorial.net/sqlite-glob/">here</a>. In a nut shell, it allows the use of placeholders like ?, *, [], [^], and ranges.
</p>

```python
import sys
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"
pattern=sys.argv[1]

vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, label=pattern)

# One wec_result for each wec specified in wec_ids
for wec_index in range(len(vecs)):
    # Index 0 element is the wec_id
    print("\nWEC: %s"%vecs[wec_index][0])
    # Index 1 element is the list of all results for this wec
    # Result list contains tuples of ([raw],[prepro],[(w,v) tuples])
    for (raw, prepro, tuples) in vecs[wec_index][1]:
        print("Raw:    '%s'"%str(raw))
        print("Prepro: %s"%str(prepro))
        for (w,v) in tuples:
            print("Unit:   %s\nVector: %s\n"%(w,str(v)))
```

<p>
Executing this code with

```shell
$ python tools/test_get_matching_vectors.py street-*
```

from the WOMBAT directory returns
</p>

<pre>
WEC: street-*@algo:glove;dataset:6b;dims:50;fold:1;unit:token
Raw:    ''
Prepro: []
Unit:   street-level
Vector: [ 0.40303  -0.7925    0.27952  -0.50839   0.19802  -0.99121   0.57987
 -0.25474   0.42316   0.71899  -0.27968  -0.65923  -0.17184   0.97444
 -0.8138   -0.69954   0.212     0.27067   0.43226  -0.43335   0.80237
 -0.27533  -0.85711   0.86139  -1.1262    0.82419   0.57705   0.76333
  0.062348  0.055458 -0.19471  -0.40467   0.91031  -0.70507  -0.23654
  0.5855    0.092822 -0.21668  -0.47821  -0.13724   0.60215  -0.32998
 -0.078702  0.61181  -0.1931   -0.58708   0.28614  -0.020059  0.2946
 -0.33449 ]

Unit:   street-legal
Vector: [-0.49192   -0.95071    0.19508    0.065616  -0.09635    0.40056
 -0.013609  -0.30365    0.57384    0.0017828  0.45458   -0.39815
  0.4951     0.24708    0.71347   -0.049488   0.079448   1.1826
  0.036627  -1.0981     0.29696   -0.66568   -0.48342    0.85139
  0.91764    0.94041   -0.55475    0.34229    1.0397     0.40414
 -0.82512   -0.27286   -0.22914    0.65506    0.40911    0.19225
  0.7302    -0.17388   -1.6934     0.046266   0.12252   -0.44143
 -0.1372    -0.75612   -0.14836   -0.25234    0.12291    0.25408
  0.23213    0.41981  ]

Unit:   street-smart
Vector: [-0.72593   -0.49009   -0.10827   -0.5269     0.17271    0.4735
  0.43434    0.61283   -0.13394    0.42256   -0.33147   -0.16036
  0.81362    0.64003   -1.0242    -0.92203    0.28986    0.70959
  0.82007   -0.0066828 -0.48355    0.34111   -0.40044    0.69414
  0.40169    0.51037   -0.083635   0.23931    0.43077   -0.21978
 -0.92982   -0.25899    0.48785    0.39913    0.30905    0.48351
 -0.40105    0.021208  -0.1318    -0.61757    0.21205   -0.30672
  0.21054   -0.10134    0.45577   -0.39337    0.44261    0.46848
  0.43588    0.75647  ]

Unit:   street-porter
Vector: [-0.96942  -0.96639  -0.60601   0.3361   -0.7169    0.06872   0.19757
  0.094712  0.30745   0.17507  -0.12757  -0.089787  1.8732    0.86427
  0.48136  -0.097871 -1.7678   -0.89827   1.3151   -0.51847  -0.34526
  0.83002   0.65794  -0.5369    0.46992   1.1141    0.27192   0.22135
 -0.79224   0.28311  -1.1193   -0.84047   0.27677  -1.5569   -0.46797
  0.17451   0.24473  -0.32943  -0.059748 -1.5041   -0.62334   0.18818
  0.23813  -0.03457   0.29596   0.47728   0.17278  -0.67522   0.13767
  0.65232 ]

Unit:   street-side
Vector: [-0.064944 -0.46447  -0.34895  -0.26038  -0.24787  -0.79229   0.13495
  0.35999   1.0211    0.4791   -0.79494  -0.73045   0.54308   0.31388
 -0.12127  -0.045159 -0.1183   -0.26375   1.0439   -0.49615   0.44488
 -0.40227  -0.72437   0.96447  -0.1821    1.5393    0.40531   0.72948
  0.82579   0.43041  -0.39022   0.028626  0.76456   0.12112   0.2589
  0.90282   0.54276  -0.043732 -0.83615   0.18182   0.10371  -0.065432
 -1.1248    0.88714  -0.23558  -0.55528   0.49327  -0.080848  0.28676
  0.18981 ]

Unit:   street-wise
Vector: [-0.3011    -0.17117   -0.15991   -0.62238    0.29494   -0.014078
  0.67835    0.05539    0.049974   0.49069   -0.21743    0.05661
  0.60609    0.84505   -0.64945   -0.70532   -0.27278    0.06957
  0.9014     0.21102   -0.9246     0.35939   -0.32785    0.43468
  0.63198    0.62846    0.20951   -0.052988   0.39349    0.24415
 -1.0438     0.22403    0.23376    0.56951    0.0081678  0.34427
 -0.57283    0.0093141 -0.24427   -0.075364  -0.11434   -0.011186
  0.23007    0.55659    0.54339   -0.59993    0.64632   -0.17135
  0.57521    0.19062  ]

Unit:   street-fighting
Vector: [-0.19546   -0.2904    -1.029      0.04422   -0.91767   -0.1598
  0.74051    0.33155    0.13802    0.86638   -0.27616   -0.26133
  0.62979   -0.16358   -0.6652    -0.68577   -0.57362   -0.13795
  0.65736    0.31993   -0.073709   0.44859    0.51312   -0.38507
  0.38218    1.2236    -0.23161   -0.6245     0.56195    0.75371
 -0.46868   -0.579      0.0064889 -0.85418    0.20035    0.79038
 -0.76457    0.85578   -0.33117    0.54323   -0.10381   -0.45369
 -0.20833    0.84081    0.84439   -0.21615    1.1818     0.67904
 -0.54561    0.21325  ]

Unit:   street-corner
Vector: [ 0.069143 -0.13195  -0.86449  -0.62174   0.18645  -0.42145   0.71741
  0.39719  -0.48795   0.446    -0.77206  -0.037595  0.39778   0.84673
 -0.91484  -0.62166  -0.2544    0.30072   1.2523   -0.046946  0.75456
 -0.060261 -0.21561   0.2985   -0.32341   1.1494    0.070652  0.28884
 -0.24121   0.29182  -0.88051   0.16276   0.17997   0.91511  -0.066837
  0.016092 -0.34234   0.03839  -0.92554  -0.01586  -0.050597 -0.9135
 -0.78377   0.55204   0.60143  -0.94121   0.11934   0.14358   0.061115
  0.17102 ]
</pre>


<h3>Integrating automatic preprocessing</h3>
<h4>Simple preprocessing (no MWEs)</h4>

<p>

In order to process raw input, WOMBAT supports the integration of arbitrary preprocessing python code right into the word embedding database. Then, if WOMBAT is accessed with the attribute ```raw=True```, this code is automatically executed in the background. 

</p>

<p>

WOMBAT provides the class ```wombat_api.preprocessors.preprocessor_stub.py``` to be used as a base for customized preprocessing code.

</p>

```python
import pickle

# Stop-word replacement
SW_SYMBOL="*sw*"

class preprocessor(object):
    def __init__(self, name=__name__, phrasefile="", verbose=False):

        if verbose: print("Initializing preprocessor %s"%name)

    """ This method is called from WOMBAT.
        'line' is the raw string to be processed,
        'unit' is the processing unit to be used (e.g. token, stem). 
    """
    def process(self, line, unit, fold=True, sw_symbol=SW_SYMBOL, conflate=False, 
                no_phrases=False, verbose=False): 

        # Lowercase if fold==True
        if fold: line=line.lower()
        # This does the most rudimentary preprocessing only
        return line.split(" ")        

    def pickle(self, picklefile):
        pickle.dump(self, open(picklefile,"wb"), protocol=pickle.HIGHEST_PROTOCOL)
```

<p>

However, WOMBAT also provides the ready-to-use standard preprocessor ```wombat_api.preprocessors.standard_preprocessor.py``` (based on NLTK 3.2.5). In order to link it (or <b>any other preprocessing code</b> based on the above stub!!) to one or more WECs in WOMBAT, a pickled instance has to be created first, and then linked to one or more WECs. 

</p>

```python
from wombat_api.preprocessors import standard_preprocessor
from wombat_api.core import connector as wb_conn

# Create instance 
prepro=standard_preprocessor.preprocessor(name="my_standard_preprocessor", phrasefile="")
prepro.pickle("temp/my_standard_preprocessor.pkl")

# Connect to WOMBAT
wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=True)
# Assign pickled instance to GloVe WECs
wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token", 
                        "temp/my_standard_preprocessor.pkl")

# Calling this method with an empty string as pickle file name removes the preprocessor.
wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token", "")


```

<p>
After that, raw, unprocessed input data can be streamed directly into WOMBAT's vector retrieval methods.
</p>

```python
import numpy as np
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"
rawfile="data/text/STS.input.track5.en-en.txt"

vecs = wbc.get_vectors(wec_ids, {}, 
            for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0)], 
            raw=True)

# One wec_result for each wec specified in wec_ids
for wec_index in range(len(vecs)):
    # Index 0 element is the wec_id
    print("\nWEC: %s"%vecs[wec_index][0])
    # Index 1 element is the list of all results for this wec
    # Result list contains tuples of ([raw],[prepro],[(w,v) tuples])
    for (raw, prepro, tuples) in vecs[wec_index][1]: 
        print("Raw:    '%s'"%str(raw))
        print("Prepro: %s"%str(prepro))
        for (w,v) in tuples:
            print("Unit:   %s\nVector: %s\n"%(w,str(v)))
```

<p>
Executing this code with

```shell
$ python tools/test_get_vectors_from_raw.py
```

from the WOMBAT directory returns (abbreviated)
</p>

<pre>
WEC: algo:glove;dataset:6b;dims:50;fold:1;unit:token
Raw:    'A person is on a baseball team.'
Prepro: ['*sw*', 'person', '*sw*', '*sw*', '*sw*', 'baseball', 'team']
Unit:   baseball
Vector: [-1.9327    1.0421   -0.78515   0.91033   0.22711  -0.62158  -1.6493
  0.07686  -0.5868    0.058831  0.35628   0.68916  -0.50598   0.70473
  1.2664   -0.40031  -0.020687  0.80863  -0.90566  -0.074054 -0.87675
 -0.6291   -0.12685   0.11524  -0.55685  -1.6826   -0.26291   0.22632
  0.713    -1.0828    2.1231    0.49869   0.066711 -0.48226  -0.17897
  0.47699   0.16384   0.16537  -0.11506  -0.15962  -0.94926  -0.42833
 -0.59457   1.3566   -0.27506   0.19918  -0.36008   0.55667  -0.70315
  0.17157 ]

Unit:   person
Vector: [ 0.61734    0.40035    0.067786  -0.34263    2.0647     0.60844
  0.32558    0.3869     0.36906    0.16553    0.0065053 -0.075674
  0.57099    0.17314    1.0142    -0.49581   -0.38152    0.49255
 -0.16737   -0.33948   -0.44405    0.77543    0.20935    0.6007
  0.86649   -1.8923    -0.37901   -0.28044    0.64214   -0.23549
  2.9358    -0.086004  -0.14327   -0.50161    0.25291   -0.065446
  0.60768    0.13984    0.018135  -0.34877    0.039985   0.07943
  0.39318    1.0562    -0.23624   -0.4194    -0.35332   -0.15234
  0.62158    0.79257  ]

Unit:   team
Vector: [-0.62801    0.12254   -0.3914     0.87937    0.28572   -0.41953
 -1.4265     0.80463   -0.27045   -0.82499    1.0277     0.18546
 -1.7605     0.18552    0.56819   -0.38555    0.61609    0.51209
 -1.5153    -0.45689   -1.1929     0.33886    0.18038    0.10788
 -0.35567   -1.5701    -0.02989   -0.38742   -0.60838   -0.59189
  2.9911     1.2022    -0.52598   -0.76941    0.63006    0.63828
  0.30773    1.0123     0.0050781 -1.0326    -0.29736   -0.77504
 -0.27015   -0.18161    0.04211    0.32169    0.018298   0.85202
  0.038442  -0.050767 ]

Raw:    'Our current vehicles will be in museums when everyone has their own aircraft.'
Prepro: ['*sw*', 'current', 'vehicles', '*sw*', '*sw*', '*sw*', 'museums', '*sw*', 'everyone', '*sw*', '*sw*', '*sw*', 'aircraft']
Unit:   aircraft
Vector: [ 1.7714    -0.75714    1.0217    -0.26717   -0.36311    0.29269
 -0.79656   -0.49746    0.41422   -1.0602     1.2215     0.41672
 -0.40249    0.70013   -1.0695    -0.19489   -1.0886     1.2409
 -2.1505    -1.1609     0.10969    0.1729    -0.82806   -0.97654
 -0.14616   -1.2641    -0.13635   -0.041624   1.0939     0.7116
  2.474     -0.16225   -0.26348    0.15532    1.1995     0.0076471
  0.76388   -0.071138  -1.3869     0.88787    0.36175   -0.33419
  1.6512    -0.52295   -0.30657    0.17399   -0.55383    0.46204
 -0.59634    0.41802  ]

Unit:   current
Vector: [-9.7534e-02  7.9739e-01  4.5293e-01  8.8687e-03 -5.1178e-02  1.8178e-02
 -1.1791e-01 -6.9793e-01 -1.5940e-01 -3.3886e-01  2.1386e-01  1.1945e-01
 -3.3078e-01  7.0846e-02  5.3858e-01  5.2766e-01 -9.7989e-02  3.4390e-02
  6.6567e-02 -2.7172e-01  1.1587e-01 -7.7042e-01 -2.3377e-01 -8.5757e-02
 -2.7538e-01 -1.2693e+00  1.5670e-01 -4.5892e-02 -3.4532e-01  1.3033e+00
  3.6207e+00  9.1328e-03 -1.2680e-01 -6.1576e-01  6.6010e-02 -2.5451e-01
  1.3535e-03 -5.1221e-02 -2.2177e-01 -4.4328e-01 -5.4152e-01  1.9691e-01
 -3.3034e-01  3.7052e-03 -8.5744e-01  1.6703e-01  4.1405e-02  5.9579e-01
 -9.7806e-02  1.8642e-01]

Unit:   everyone
Vector: [ 4.7246e-02  4.2534e-02  1.1150e-01 -5.3334e-01  1.1487e+00 -4.1835e-01
 -4.1667e-01  4.6632e-01 -3.9396e-02  2.1353e-01 -1.6719e-01  2.3585e-01
 -3.4603e-01 -3.8585e-02  1.0645e+00  4.6839e-01  4.4521e-01  3.3946e-01
  2.9733e-01 -9.3541e-01 -2.7267e-01  9.1747e-01 -2.6640e-02  4.9671e-01
  1.2452e+00 -1.8388e+00 -5.4239e-01  4.7746e-01  9.3603e-01 -9.2198e-01
  2.7160e+00  1.1366e+00 -2.2590e-01 -3.8464e-01 -6.0182e-01 -2.2687e-01
  1.1669e-01  3.2993e-02  2.3049e-01 -4.9548e-01 -2.5239e-01  6.3638e-02
 -8.7472e-02  5.5913e-01 -7.1459e-05  2.4938e-01 -2.1032e-01 -2.3587e-01
 -1.0124e-01  7.5840e-01]

Unit:   museums
Vector: [ 9.8518e-01  1.1344e+00 -6.2976e-01 -3.3453e-01  3.5321e-02 -1.2801e+00
 -1.0494e+00 -6.9263e-01 -1.5120e-02 -6.1263e-02 -1.9171e-01 -1.3570e-03
  5.4254e-01  1.7061e-01  5.3629e-01  3.4711e-02  8.7502e-01  4.1138e-03
 -4.1096e-02  7.3491e-02  1.2865e+00 -2.0661e-01 -8.3286e-01  3.6639e-01
 -6.3374e-01 -2.2028e-01 -1.3518e+00 -3.8629e-01 -5.3463e-01 -1.2197e+00
  1.5524e+00  6.9474e-01  1.0281e+00 -1.5287e+00 -5.2155e-01  8.3129e-01
  8.5204e-02  8.9238e-01 -4.5974e-01  5.4429e-01  1.5087e-01 -6.4565e-01
  1.7007e+00  6.5024e-01 -1.6995e-01  9.4863e-01 -1.0720e+00  7.9241e-02
 -5.7654e-01 -7.3065e-01]

Unit:   vehicles
Vector: [ 0.75982  -0.76559   2.0944   -0.37478  -0.34947   0.18489  -1.1152
 -1.0155    0.24493  -0.71603   0.60359  -1.0472   -0.28302  -0.36222
  0.29956   0.043537 -0.31847   1.4753   -0.49762  -2.1802    0.52873
 -0.3492   -0.7874   -0.058825 -0.11986  -0.59238  -0.19368   0.42545
  1.2132    0.19446   2.6633    0.30815  -0.1981   -0.28798   1.1756
  0.682     0.4655   -0.3504   -1.0034    0.83025  -0.2051   -0.24585
  1.1062   -0.8197    0.26461  -0.73376  -0.53285   0.035146  0.25134
 -0.60158 ]

- - - - - - - - - - - - - - - - - - cut - - - - - - - - - - - - - - - -

</pre>

<h4>Advanced preprocessing with MWEs</h4>

<p>
Preprocessing raw textual data for embedding vector lookup becomes non-trivial when the <b>WEC training data</b> itself was processed in a non-trivial way: When the training data was <b>stemmed</b>, the WEC <b>vocabulary</b> also consists of stems, and turning raw textual data into compatible units for lookup requires -- ideally -- that the exact same stemming algorithm be applied to it. The same is true for any other word-level normalization / modification that might have been applied to the WEC training data. Integrating preprocessing code into embedding vector lookup, as described above, is a first step towards acknowledging the importance of preprocessing.
</p>

<p>
For pretrained WECs, like GloVe above, the preprocessing code is often not available, or preprocessing is considered trivial. In these cases, it is possible with reasonable effort to inspect the WEC vocabulary and derive preprocessing rules which more or less imitate the original preprocessing. The standard_preprocessor class used above is an example of this.
</p>

<p>

Preprocessing code to be integrated into WOMBAT supports an optional ```phrasespotter.py``` module, which can be initialized with a list of phrases / multi-word expressions that you want to be treated as tokens. For custom, self-trained WECs, the procedure is ideally the following:
<ol>
<li>Obtain a list or dictionary of phrases / multi-word expressions. This can either be a preexisting, manually curated resource (e.g. based on the <a href="https://cso.kmi.open.ac.uk/downloads">Computer Science Ontology</a>), or a list of phrases mined automatically from some text (e.g. with <a href="http://elkishk2.web.engr.illinois.edu/">ToPMine</a>).</li>
<li>Create a preprocessor as above, providing the name of the file containing the phrases (one per line) as value to the phrasefile parameter.

```python
from wombat_api.preprocessors import standard_preprocessor

prepro=standard_preprocessor.preprocessor(name="my_cs_savvy_standard_preprocessor", 
                                          phrasefile="data/mwes/cso-mwes-stemmed.txt")
prepro.pickle("temp/my_cs_savvy_standard_preprocessor.pkl")

```
</li>
<li>

Apply the preprocessor to the raw WEC training data <b>before training the WECs</b>. WOMBAT provides the script ```tools/apply_preprocessor.py``` for that purpose.

```shell
$ python tools/apply_preprocessor.py data/text/dblp-titles.txt temp/my_cs_savvy_standard_preprocessor.pkl
```
The script produces the following output

```shell
data/text/dblp-titles.txt.conf.

```

</li>
</ol>
</p>

