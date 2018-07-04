# WOMBAT
Word Embedding Database

<b>Under construction!</b>

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

<b>WOMBAT</b>, the WOrd eMBedding dATabase, is a light-weight Python tool for more transparent, efficient, and robust access to potentially large numbers of word embedding collections (WECs). 
It supports NLP researchers and practitioners in developing compact, efficient, and reusable code. 
Key features of WOMBAT are
<ol type=1>
<li>transparent identification of WECs by means of a clean syntax and human-readable features, </li>
<li>efficient lazy, on-demand retrieval of word vectors, and</li> 
<li>increased robustness by systematic integration of executable preprocessing code. </li>
</ol>

WOMBAT implements some Best Practices for research reproducibility and complements existing approaches towards WEC standardization and sharing. 
WOMBAT provides a single point of access to existing WECs. Each plain text WEC file has to be imorted into WOMBAT once, receiving in the process a set of ATT:VAL identifiers consisting of five system attributes (algo, dims, dataset, unit, fold) plus arbitrarily many user-defined ones.

<h3>Importing Pre-Trained Embeddings to WOMBAT: GloVe</h3>
One of the main uses of WOMBAT is as a wrapper for accessing existing, off-the-shelf word embeddings like e.g. GloVe. (The other use involves access to self-trained embeddings, including preprocessing and handling of multi-word-expressions, cf. below.)

The following code is sufficient to import a sub set of the GloVe embeddings. 
<pre>
from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
importpath="data/embeddings/"
wbc = wb_conn(path=wbpath, create_if_missing=True)
for d in ['50', '100', '200', '300']:
    wbc.import_from_file(importpath+"glove.6B."+d+"d.txt", 
                         "algo:glove;dataset:6b;dims:"+d+";fold:1;unit:token")
</pre>

<p>
To execute this code, run 
<pre>
python tools/import_to_wombat.py
</pre>
from the WOMBAT directory.
</p>

<p>
The required GloVe embeddings are <b>not part of WOMBAT</b> and can be obtained from Stanford <a href="http://nlp.stanford.edu/data/wordvecs/glove.6B.zip">here</a>. Extract them into "data/embeddings/".
The WOMBAT master and embeddings data bases will be created in "data/wombat-data". 
</p>
<p>
The above import assigns only the minimally required ATT:VAL pairs to the embeddings.
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

<pre>
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"

vecs = wbc.get_vectors(wec_ids, {}, for_input=[['this','is','a', 'test'], ['yet', 'another', 'test']])

# One wec_result for each wec specified in wec_identifier
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
</pre>

<p>
To execute this code, run 
<pre>
python tools/test_get_vectors.py
</pre>
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
The follwing code looks up embedding vectors matching the supplied pattern. The pattern uses the GLOB syntax described <a href=http://www.sqlitetutorial.net/sqlite-glob/">here</a>. In a nut shell, it allows the use of ?, *, [], [^], and ranges.
</p>
