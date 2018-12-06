# WOMBAT
<p>

See <a href="http://arxiv.org/abs/1807.00717" target="_blank">this paper</a> (to appear at <a href="http://coling2018.org/accepted-demo-papers/" target="_blank">COLING 2018</a>) for additional details. Please cite the COLING paper if you use WOMBAT.
</p>

<p>
Note: Due to a name clash with another python package, the actual WOMBAT package structure is slightly different than that used in the COLING paper examples! The examples used in this web site are up-to-date.
</p>

<!-- <img src="https://github.com/nlpAThits/WOMBAT/blob/master/data/images/wombat.png" height="300px" align="middle" /> -->

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
<ol type="1">
<li>transparent identification of WECs by means of a clean syntax and human-readable features, </li>
<li>efficient lazy, on-demand retrieval of word vectors, and</li> 
<li>increased robustness by systematic integration of executable preprocessing code. </li>
</ol>
</p>

<p>
WOMBAT implements some Best Practices for research reproducibility and complements existing approaches towards WEC standardization and sharing. 

WOMBAT provides a single point of access to existing WECs. Each plain text WEC file has to be imported into WOMBAT once, receiving in the process a set of ```ATT:VAL``` identifiers consisting of five system attributes (algo, dims, dataset, unit, fold) plus arbitrarily many user-defined ones.

</p>


<ol>

<!--<li> 
[**Introduction**](#introduction) 
</li>
-->
<li> 

[**Installation**](#installation)

</li> 
<li> 

[**Importing Pre-Trained Embeddings to WOMBAT: GloVe**](#importing-pre-trained-embeddings-to-wombat-glove)

</li> 
<li> 

[**Integrating automatic preprocessing**](#integrating-automatic-preprocessing)

<ol type="1">

<li> 

[**Simple preprocessing**](#simple-preprocessing-no-mwes)

</li> 
<li> 

[**Advanced preprocessing with MWE)**](#advanced-preprocessing-with-mwes)

</li> 

</ol>

</li> 


<li> 

[**Use Cases**](#use-cases)

<ol type="1">

<li> 

[**Pairwise Distance**](#pairwise-distance)

</li> 

</ol>

</li> 


</ol>




<h3>Installation</h3>
<p>

WOMBAT does not have a lot of special requirements. 
The basic functionality only requires sqlite3, numpy, and tqdm, the ```analyse``` module requires psutil, matplotlib, and scikit-learn in addition. Note that sqlite3 is commonly available as a default package, e.g. with conda.

</p>

<p>
In addition, the standard_preprocessor (see below) requires NLTK 3.2.5. 
A working environment can be set up like this:
</p>

<p>

```shell
$ conda create --name wombat python=3.6 numpy tqdm psutil matplotlib scikit-learn nltk==3.2.5
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

One of the main uses of WOMBAT is as a wrapper for accessing existing, off-the-shelf word embeddings like e.g. GloVe. (The other use involves access to self-trained embeddings, including preprocessing and handling of multi-word-expressions, cf. [below](#integrating-automatic-preprocessing))

The following code is sufficient to import a sub set of the GloVe embeddings. 
```python
from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
importpath="data/embeddings/glove.6B/"

wbc = wb_conn(path=wbpath, create_if_missing=True)

for d in ['50', '100', '200', '300']:
    for n in ['none', 'abtt']:
        wbc.import_from_file(importpath+"glove.6B."+d+"d.txt", 
                             "algo:glove;dataset:6b;dims:"+d+";fold:1;unit:token;norm:"+n, 
                             normalize=n, 
                             prepro_picklefile="")
```
Using ```norm:abtt``` (<i>"All but the top"</i>) creates a normalized version as described in <a href="https://arxiv.org/abs/1702.01417" target=_new>this</a> paper. Parameter D is set to ```D=max(int(dim/100), 1)```.
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

The above import assigns the following <b>minimally required</b> system ```ATT:VAL``` pairs to the embeddings.

<table>
<tr><td><b>Attribute</b></td><td><b>Meaning</b></td></tr>
<tr><td>algo</td><td>Descriptive label for the <b>algorithm</b> used for training these embeddding vectors.</td></tr>
<tr><td>dataset</td><td>Descriptive label for the <b>data set</b> used for training these embedding vectors.</td></tr>
<tr><td>dims</td><td><b>Dimensionality</b> of these embedding vectors. Required for description and for creating right-sized <b>empty</b> vectors for OOV words.</td></tr>
<tr><td>fold</td><td>Indicates whether the embedding vectors are <b>case-sensitive</b> (fold=0) or not (fold=1). If fold=1, input words are lowercased before lookup.</td></tr>
<tr><td>unit</td><td>Unit of representation used in the embedding vectors. Works as a descriptive label with pre-trained embeddings for which no custom preprocessing has been integrated into WOMBAT. If custom preprocessing exists, the value of this attribute is passed to the process() method. The current preprocessor modules (cf. below) support the values <b>stem</b> and <b>token</b>.</td></tr>
</table>

In addition, the following user-defined ```ATT:VAL``` pair is assigned.
<table>
<tr><td><b>Attribute</b></td><td><b>Meaning</b></td></tr>
<tr><td>norm</td><td>Descriptive label for the <b>normalization</b> applied at input time. <b>none</b> or one of <b>l1</b>, <b>l2</b>, or <b>abtt</b>.</td></tr>
</table>

</p>

<p>
After import, the embedding vectors are immediately available for efficient lookup of <b>already preprocessed</b> words. 
The following code accesses two of the eight GloVe WECs and looks up &lt;unit, vector&gt; tuples for two sequences of words. For performance reasons, input order is ignored.
</p>

```python
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token;norm:{none,abtt}"

vecs = wbc.get_vectors(wec_ids, {}, 
                       for_input=[['this','is','a', 'test'], ['yet', 'another', 'test']], 
                       in_order=False)

# One wec_result for each wec specified in wec_identifier. norm:{none,abtt} notation is expanded at execution time.
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
The result is a nested python list with one result set for each supplied WEC identifier.
</p>

<pre>
WEC: algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
Raw:    ''
Prepro: ['this', 'is', 'a', 'test']
Unit:   a
Vector: [ 0.21705     0.46515    -0.46757001  0.10082     1.01349998  0.74844998
 -0.53104001 -0.26256001  0.16812     0.13181999 -0.24909    -0.44185001
 -0.21739     0.51003999  0.13448    -0.43141001 -0.03123     0.20674001
 -0.78138    -0.20148    -0.097401    0.16088    -0.61835998 -0.18504
 -0.12461    -2.25259995 -0.22321001  0.5043      0.32257     0.15312999
  3.96359992 -0.71364999 -0.67012     0.28388     0.21738     0.14432999
  0.25926     0.23434     0.42739999 -0.44451001  0.13812999  0.36973
 -0.64288998  0.024142   -0.039315   -0.26036999  0.12017    -0.043782
  0.41012999  0.1796    ]

Unit:   is
Vector: [  6.18499994e-01   6.42539978e-01  -4.65519994e-01   3.75699997e-01
   7.48380005e-01   5.37389994e-01   2.22390005e-03  -6.05769992e-01
   2.64079988e-01   1.17030002e-01   4.37220007e-01   2.00920001e-01
  -5.78589998e-02  -3.45889986e-01   2.16639996e-01   5.85730016e-01
   5.39189994e-01   6.94899976e-01  -1.56179994e-01   5.58300018e-02
  -6.05149984e-01  -2.89970011e-01  -2.55939998e-02   5.55930018e-01
   2.53560007e-01  -1.96120000e+00  -5.13809979e-01   6.90959990e-01
   6.62460029e-02  -5.42239994e-02   3.78710008e+00  -7.74030030e-01
  -1.26890004e-01  -5.14649987e-01   6.67050034e-02  -3.29329997e-01
   1.34829998e-01   1.90490007e-01   1.38119996e-01  -2.15030000e-01
  -1.65730007e-02   3.12000006e-01  -3.31889987e-01  -2.60010008e-02
  -3.82030010e-01   1.94030002e-01  -1.24660000e-01  -2.75570005e-01
   3.08990002e-01   4.84970003e-01]

Unit:   test
Vector: [ 0.13175    -0.25516999 -0.067915    0.26192999 -0.26155001  0.23569
  0.13077    -0.011801    1.76590002  0.20781     0.26198    -0.16428
 -0.84641999  0.020094    0.070176    0.39778     0.15278    -0.20213
 -1.61839998 -0.54326999 -0.17856     0.53894001  0.49868    -0.10171
  0.66264999 -1.70510006  0.057193   -0.32405001 -0.66834998  0.26653999
  2.84200001  0.26844001 -0.59536999 -0.50040001  1.51989996  0.039641
  1.66589999  0.99757999 -0.55970001 -0.70493001 -0.0309     -0.28301999
 -0.13564     0.64289999  0.41490999  1.23619998  0.76586998  0.97798002
  0.58507001 -0.30175999]

Unit:   this
Vector: [  5.30740023e-01   4.01169986e-01  -4.07849997e-01   1.54440001e-01
   4.77820009e-01   2.07540005e-01  -2.69510001e-01  -3.40229988e-01
  -1.08790003e-01   1.05630003e-01  -1.02890000e-01   1.08489998e-01
  -4.96809989e-01  -2.51280010e-01   8.40250015e-01   3.89490008e-01
   3.22840005e-01  -2.27970004e-01  -4.43419993e-01  -3.16489995e-01
  -1.24059997e-01  -2.81699985e-01   1.94670007e-01   5.55129983e-02
   5.67049980e-01  -1.74189997e+00  -9.11450028e-01   2.70359993e-01
   4.19270009e-01   2.02789996e-02   4.04050016e+00  -2.49430001e-01
  -2.04160005e-01  -6.27619982e-01  -5.47830015e-02  -2.68830001e-01
   1.84440002e-01   1.82040006e-01  -2.35359997e-01  -1.61550000e-01
  -2.76549995e-01   3.55059989e-02  -3.82110000e-01  -7.51340005e-04
  -2.48219997e-01   2.81639993e-01   1.28189996e-01   2.87620008e-01
   1.44400001e-01   2.36110002e-01]

Raw:    ''
Prepro: ['yet', 'another', 'test']
Unit:   another
Vector: [ 0.50759     0.26321     0.19638     0.18407001  0.90792     0.45267001
 -0.54491001  0.41815999  0.039569    0.061854   -0.24574    -0.38501999
 -0.39649001  0.32165     0.59610999 -0.39969999 -0.015734    0.074218
 -0.83148003 -0.019284   -0.21331     0.12873    -0.25409999  0.079348
  0.12588    -2.12940001 -0.29091999  0.044597    0.27353999 -0.037492
  3.45799994 -0.34641999 -0.32802999  0.17566     0.22466999  0.08987
  0.24528     0.070129    0.2165     -0.44312999  0.02516     0.40817001
 -0.33533001  0.0067758   0.11499    -0.15701    -0.085219    0.018568
  0.26124999  0.015387  ]

Unit:   test
Vector: [ 0.13175    -0.25516999 -0.067915    0.26192999 -0.26155001  0.23569
  0.13077    -0.011801    1.76590002  0.20781     0.26198    -0.16428
 -0.84641999  0.020094    0.070176    0.39778     0.15278    -0.20213
 -1.61839998 -0.54326999 -0.17856     0.53894001  0.49868    -0.10171
  0.66264999 -1.70510006  0.057193   -0.32405001 -0.66834998  0.26653999
  2.84200001  0.26844001 -0.59536999 -0.50040001  1.51989996  0.039641
  1.66589999  0.99757999 -0.55970001 -0.70493001 -0.0309     -0.28301999
 -0.13564     0.64289999  0.41490999  1.23619998  0.76586998  0.97798002
  0.58507001 -0.30175999]

Unit:   yet
Vector: [ 0.69349998 -0.13891999 -0.10862    -0.18671     0.56310999  0.070388
 -0.52788001  0.35681    -0.21765     0.44887999 -0.14023     0.020312
 -0.44203001  0.072964    0.85846001  0.41819     0.19097    -0.33511999
  0.012309   -0.53561002 -0.44547999  0.38117     0.2255     -0.26947999
  0.56835002 -1.71700001 -0.76059997  0.43305999  0.41890001  0.091699
  3.2262001  -0.18561    -0.014535   -0.69815999  0.21151    -0.28681999
  0.12492     0.49278    -0.57783997 -0.75677001 -0.47876    -0.083749
 -0.013377    0.19862001 -0.14819001  0.21787    -0.30472001  0.54255003
 -0.20916     0.14964999]


WEC: algo:glove;dataset:6b;dims:50;fold:1;norm:abtt;unit:token
Raw:    ''
Prepro: ['this', 'is', 'a', 'test']
Unit:   a
Vector: [-0.38456726  0.39097878 -0.1628997   0.35068694  0.99550414  0.44776174
 -0.50116265  0.31360865  0.35520661 -0.12043196 -0.06741576  0.22319981
 -0.3842575   0.31569615  0.12704191 -0.6358701   0.36765504 -0.2414223
  0.2757951  -0.06014517 -0.47552517  0.17220016 -0.76332432 -0.32266825
  0.3489612  -1.037853    0.32191628  0.15478981 -0.11307254  0.47718403
  1.48160338 -1.41211295 -0.17363971  0.33394873 -0.05526268  0.04968219
  0.40862644  0.32090271  0.75181049  0.07840931  0.39596623  0.88622624
 -0.85963786 -0.91397953  0.53625643 -0.70439553 -0.31108141 -0.22278789
  0.51454931  1.25660634]

Unit:   is
Vector: [ 0.09126818  0.60529983 -0.19061366  0.60591251  0.75278735  0.27584556
 -0.00489476 -0.10457748  0.42818767 -0.12769794  0.5956223   0.79856926
 -0.23736086 -0.52514869  0.23125611  0.40881187  0.9044193   0.28455088
  0.76149231  0.16461219 -0.9323107  -0.26970825 -0.14817345  0.42578259
  0.66423047 -0.9320755  -0.04194349  0.37159386 -0.32375848  0.23331042
  1.64041948 -1.39662826  0.2985028  -0.49035078 -0.17418115 -0.42143601
  0.27057451  0.27170798  0.43615541  0.24219584  0.20077799  0.79368269
 -0.51842153 -0.87728345  0.13601783 -0.19085133 -0.53250313 -0.44660494
  0.4021166   1.45063889]

Unit:   test
Vector: [-0.38716662 -0.2882818   0.20366421  0.48994547 -0.25463828 -0.02147874
  0.11951575  0.48101032  1.92743909 -0.03607689  0.41778082  0.42583492
 -1.02733421 -0.15747839  0.08725743  0.22394061  0.51424712 -0.60825217
 -0.71632195 -0.43812668 -0.50002372  0.56020129  0.37860283 -0.2310212
  1.06628919 -0.69672549  0.52087027 -0.64004648 -1.05325282  0.54999208
  0.73280215 -0.34567764 -0.17792362 -0.47898144  1.28256369 -0.05218088
  1.80012178  1.07820046 -0.26461291 -0.25504762  0.18192536  0.19477114
 -0.31879386 -0.19867522  0.92652762  0.85793     0.36064354  0.80783612
  0.67693424  0.65146315]

Unit:   this
Vector: [-0.05769843  0.33354184 -0.10845295  0.40082479  0.46379334 -0.08621311
 -0.24618721  0.22265518  0.07422543 -0.14528893  0.07466117  0.76159853
 -0.66591597 -0.4429512   0.83671933  0.18990952  0.71576232 -0.669433
  0.58903944 -0.18092248 -0.49315494 -0.26879567  0.0536716  -0.08078988
  1.02947712 -0.56003988 -0.3793031  -0.07380959 -0.00828686  0.33786285
  1.6179111  -0.93445206  0.27972579 -0.58211684 -0.3217994  -0.36302748
  0.33139306  0.26765579  0.08437763  0.34973046 -0.02588651  0.54583436
 -0.59350443 -0.92348766  0.31716001 -0.15190703 -0.29891419  0.11002633
  0.24681857  1.29339063]

Raw:    ''
Prepro: ['yet', 'another', 'test']
Unit:   another
Vector: [-0.025814    0.22290546  0.47375607  0.41591337  0.91046846  0.1878776
 -0.5489589   0.92557371  0.20558342 -0.18349826 -0.08540669  0.21822196
 -0.57494354  0.1411396   0.60889614 -0.5789035   0.35228795 -0.33926874
  0.09776771  0.0921993  -0.54469943  0.1482498  -0.37853688 -0.05142017
  0.54176974 -1.0848732   0.18702543 -0.27727041 -0.12025136  0.25307268
  1.2834959  -0.97531319  0.10326141  0.20209748 -0.01885122 -0.00244692
  0.38215482  0.15179047  0.51672393  0.01954687  0.24587034  0.89274144
 -0.52436882 -0.85171229  0.63781095 -0.54679894 -0.49500448 -0.15312833
  0.3553136   0.99029428]

Unit:   test
Vector: [-0.38716662 -0.2882818   0.20366421  0.48994547 -0.25463828 -0.02147874
  0.11951575  0.48101032  1.92743909 -0.03607689  0.41778082  0.42583492
 -1.02733421 -0.15747839  0.08725743  0.22394061  0.51424712 -0.60825217
 -0.71632195 -0.43812668 -0.50002372  0.56020129  0.37860283 -0.2310212
  1.06628919 -0.69672549  0.52087027 -0.64004648 -1.05325282  0.54999208
  0.73280215 -0.34567764 -0.17792362 -0.47898144  1.28256369 -0.05218088
  1.80012178  1.07820046 -0.26461291 -0.25504762  0.18192536  0.19477114
 -0.31879386 -0.19867522  0.92652762  0.85793     0.36064354  0.80783612
  0.67693424  0.65146315]

Unit:   yet
Vector: [ 0.19308138 -0.16284789  0.15555759  0.03641786  0.57559294 -0.17704657
 -0.5483343   0.83097643 -0.06182532  0.20686415  0.00978364  0.59366596
 -0.62608606 -0.10085706  0.88102579  0.25119966  0.54406774 -0.73183894
  0.87969595 -0.4385618  -0.75427032  0.40465489  0.11098945 -0.39693087
  0.95634723 -0.75478542 -0.31514072  0.12455961  0.04534632  0.3660695
  1.20038748 -0.78086185  0.38523355 -0.6831497  -0.01792914 -0.3780098
  0.25575435  0.57207143 -0.28931174 -0.32322413 -0.27600241  0.38538483
 -0.18901677 -0.6213603   0.34912282 -0.14569211 -0.7041254   0.37438834
 -0.12010401  1.07518613]

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

wec_ids=sys.argv[1]
pattern=sys.argv[2]

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
$ python tools/test_get_matching_vectors_from_wec.py "algo:glove;dataset:6b;dims:50;fold:1;unit:token" street-*
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

However, WOMBAT also provides the ready-to-use standard preprocessor ```wombat_api.preprocessors.standard_preprocessor.py``` (based on NLTK 3.2.5). In order to link it (or <b>any other preprocessing code</b> based on the above stub!!) to one or more WECs in WOMBAT, a pickled instance has to be created first, and then linked to one or more WECs. The following code is available in ```tools/assign_preprocessor.py```

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
# wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token", "")


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

We provide a plain text file of CS publication titles from the <a href="https://dblp.org">DBLP</a> site <a href="http://cosyne.h-its.org/dblp/dblp-titles.txt.tar.gz">here</a>.

Unzip it to ``` data/text/dblp-titles.txt```. 

```
Parallel Integer Sorting and Simulation Amongst CRCW Models.
Pattern Matching in Trees and Nets.
NP-complete Problems Simplified on Tree Schemas.
On the Power of Chain Rules in Context Free Grammars.
Schnelle Multiplikation von Polynomen über Körpern der Charakteristik 2.
A characterization of rational D0L power series.
The Derivation of Systolic Implementations of Programs.
Fifo Nets Without Order Deadlock.
On the Complementation Rule for Multivalued Dependencies in Database Relations.
Equational weighted tree transformations.
```


Using this data set as input, the script can be called like this:

```shell
$ python tools/apply_preprocessor.py data/text/dblp-titles.txt 
                                     temp/my_cs_savvy_standard_preprocessor.pkl
                                     stopwords:*sws* 
                                     conflate 
                                     unit:stem 
                                     fold
                                     repeat_phrases
```
to produce the following output:

```shell
data/text/dblp-titles.txt.conflated_sys.nophrases.stem
data/text/dblp-titles.txt.conflated_sys.repeat_phrases.stem
data/text/dblp-titles.txt.conflated_sys.nophrases.stem.idf
data/text/dblp-titles.txt.conflated_sys.repeat_phrases.stem.idf

```

```data/text/dblp-titles.txt.conflated_sys.nophrases.stem``` contains the plain, stemmed version of the input files:

```
parallel integ sort *sw* simul amongst crcw model 
pattern match *sw* tree *sw* net 
np complet problem simplifi *sw* tree schema 
*sw* power *sw* chain rule *sw* context free grammar 
schnell multiplik von polynomen über körpern der charakteristik 0 
*sw* character *sw* ration d0l power seri 
*sw* deriv *sw* systol implement *sw* program 
fifo net without order deadlock 
*sw* complement rule *sw* multivalu depend *sw* databas relat 
equat weight tree transform 
```

```data/text/dblp-titles.txt.conflated_sys.repeated_phrases.stem``` contains the stemmed version of the input files, with identified phrases.
In addition, due to the ```repeat_phrases``` switch, it contains a plain copy of each line in which at least one phrase was detected.

```
parallel integ sort *sw* simul amongst crcw model 
pattern_match *sw* tree *sw* net 
pattern match *sw* tree *sw* net 
np complet problem simplifi *sw* tree schema 
*sw* power *sw* chain rule *sw* context_free_grammar 
*sw* power *sw* chain rule *sw* context free grammar 
schnell multiplik von polynomen über körpern der charakteristik 0 
*sw* character *sw* ration d0l power seri 
*sw* deriv *sw* systol implement *sw* program 
fifo net without order deadlock 
*sw* complement rule *sw* multivalu depend *sw* databas relat 
equat weight tree transform 
```

```data/text/dblp-titles.txt.conflated_sys.repeated_phrases.stem.idf``` contains idf scores for all vocabulary items.


```
parallel	5.9009944474123
integ	8.105335037869118
sort	8.476328191481095
*sw*	1.8121353984487958
simul	5.7200901939963575
amongst	11.67999918619934
crcw	13.33225709581637
model	4.221747418292076
pattern_match	9.385228981189533
tree	6.3878685829354325
net	7.425108697454633
pattern	6.269503282251706
match	6.71239224432375
np	9.158831826956924
complet	7.385855293345302
problem	5.400074426355499
simplifi	8.818311696228356
schema	8.479982721069225
power	5.880688809116575
chain	7.260870040566218
rule	6.757268427774883
context_free_grammar	10.561623408412391
context	6.646591236440547
free	6.905869776159018
grammar	7.980991554950237
```
</li>

<li>
Train embedding vectors on the preprocessed training data, using your favourite training algorithm and setup.
</li>

<li>

[Import](#importing-pre-trained-embeddings-to-wombat-glove) the embedding vectors into WOMBAT, and [assign the preprocessor](#integrating-automatic-preprocessing), using the code above.

</li>

<li>
<b>Done!</b> You are all set now to retrieve embedding vectors for arbitrary, raw input text, and <b>fast</b>!!
</li>

</ol>
</p>

<h3>Use Cases</h3>
<h4>Pairwise Distance</h4>

<p>

The computation of pairwise semantic distance is a standard task in NLP. One common application is computing the <b>similarity of pre-defined sentence pairs</b>. WOMBAT provides the script ```tools/sentence_pair_similarity.py``` for this task, which uses the method ```wombat_api.analyse.plot_pairwise_distances```.

 ```python
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
vecs1 = wbc.get_vectors(wec_ids, pp_cache, 
            for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0, skiprows=0)], raw=True)
vecs2 = wbc.get_vectors(wec_ids, pp_cache, 
            for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=1, skiprows=0)], raw=True)
# Use ignore_identical=True to ignore pairs whose avg. vectors are identical (=max. similarity or min. distance)
pd = plot_pairwise_distances(vecs1, vecs2, arrange_by=wec_ids, 
            pdf_name="temp/sent_sim.pdf", size=(25,10), max_pairs=20, ignore_identical=False)
```

</p>

<p>
Calling this script produces the following output:

![Wombat sentence similarity plot](https://github.com/nlpAThits/WOMBAT/blob/master/data/images/wombat_sentence_similarity.png)

</p>

<p>

One might also be interested in finding maximally similar pairs of sentences in a plain list. WOMBAT provides the script ```tools/full_pairwise_similarity.py``` for this. The main difference to the above script is that it supplies ```None``` as the value for the second parameter. This causes the ```wombat_api.analyse.plot_pairwise_distances``` method to create a <b>cartesian product</b> of all sentences supplied as value to the first, obligatory parameter.

```python
import numpy as np, scipy.spatial.distance
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import plot_pairwise_distances

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token"
rawfile="data/text/STS.input.track5.en-en.txt"

vecs1 = wbc.get_vectors(wec_ids, {}, 
            for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0, skiprows=0)], raw=True)
# Use ignore_identical=True to ignore pairs whose avg. vectors are identical (=max. similarity or min. distance)
pd = plot_pairwise_distances(vecs1, None, arrange_by=wec_ids,
            pdf_name="temp/full_pw_sim.pdf", size=(25,10), max_pairs=20, ignore_identical=False)

```

</p>

<p>

Calling this script produces the following output:

![Wombat full list similarity plot](https://github.com/nlpAThits/WOMBAT/blob/master/data/images/wombat_full_pair_similarity.png)

</p>
