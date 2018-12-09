# WOMBAT
<p>

See our <a href="http://aclweb.org/anthology/C18-2012" target="new">COLING 2018 demo paper</a> for additional details. Please cite the paper if you use WOMBAT.
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
 |        \\\`"^" ` :    ;           |   This is WOMBAT, the WOrd eMBedding dATa base API (Version 2.1)
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
$ pip install .
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

# One wec_result for each wec specified in wec_identifier. 
# norm:{none,abtt} notation is expanded at execution time.
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
from wombat_api.analyse import plot_tsne

pattern,exclude_pattern,wbpath,wec_ids="","","",""
plot=False
for i in range(len(sys.argv)):
    if sys.argv[i]=="-p":
        pattern=sys.argv[i+1]
    elif sys.argv[i]=="-xp":
        exclude_pattern=sys.argv[i+1]
    elif sys.argv[i]=="-wbpath":
        wbpath=sys.argv[i+1]
    elif sys.argv[i]=="-wecs":
        wec_ids=sys.argv[i+1]
    elif sys.argv[i]=="-plot":
        plot=True
        
wbc = wb_conn(path=wbpath, create_if_missing=False)
vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, exclude_pattern=exclude_pattern)
if plot:
    plot_tsne(vecs, iters=1000, fontsize=5, size=(10,10), arrange_by=wec_ids, silent=False)
else:
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
                
```

<p>
Executing this code with

```shell
$ python tools/test_get_matching_vectors.py -wbpath "data/wombat-data/" -wecs "algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token" -p "*comput*" -xp "*_*" 
```

from the WOMBAT directory returns from the GloVe embeddings a list of tuples for all words matching the substring <b>comput</b>, but excluding those with an underscore.
</p>

<pre>
WEC: P:*computer*;XP:*_*;@algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
Raw:    ''
Prepro: []
Unit:   computer
Vector: [ 0.079084   -0.81503999  1.79009998  0.91653001  0.10797    -0.55628002
 -0.84426999 -1.49510002  0.13417999  0.63626999  0.35146001  0.25813001
 -0.55028999  0.51055998  0.37408999  0.12092    -1.61660004  0.83653003
  0.14202    -0.52348     0.73452997  0.12207    -0.49079001  0.32532999
  0.45306    -1.58500004 -0.63848001 -1.00530005  0.10454    -0.42984
  3.18099999 -0.62186998  0.16819    -1.01390004  0.064058    0.57844001
 -0.45559999  0.73782998  0.37202999 -0.57722002  0.66441     0.055129
  0.037891    1.32749999  0.30991     0.50696999  1.23570001  0.1274
 -0.11434     0.20709001]

Unit:   computers
Vector: [ 0.56105    -1.19659996  2.4124999   0.35547999 -0.046729   -0.73904002
 -0.70042002 -1.65859997 -0.030509    0.63224     0.40307     0.30063
 -0.13483     0.20847     0.38823     0.50260001 -1.83519995  0.83701003
  0.6455     -0.72898     0.69954002 -0.21853    -0.063499    0.34255999
  0.65038002 -1.11230004 -0.41428    -1.12329996  0.62655002 -0.60872
  2.81030011  0.19251999  0.19487    -0.71785003  0.21378     0.75274003
 -0.27748001  0.81586999 -0.24152     0.040814    0.40838999 -0.0029812
  0.35493001  1.46300006  0.17201     0.80510002  0.49981999 -0.15800001
 -0.26460999 -0.38896999]

Unit:   computing
Vector: [-0.075077   -0.10027     1.18130004  0.95204997  0.041338   -0.79659998
 -0.03967    -1.66919994  0.34807     0.42230001  0.26225001  0.07144
 -0.052628   -0.041547   -0.67650998  0.0065369  -0.49070001  1.26110005
  0.64635003 -0.5262      0.21816    -0.52133    -0.44356999  0.15283
  0.55921    -0.15716    -0.68899    -1.22010005  0.040493    0.65311998
  2.38890004 -0.50182003 -0.26547    -1.20449996 -0.43509001  0.36212999
 -0.99496001  1.25100005  0.45027     0.019758    0.76959002 -0.48109999
 -0.90126997  1.56589997 -0.29357001  0.32879999  1.13759995  0.15703
 -0.20730001  0.50344002]

Unit:   computerized
Vector: [ 0.22301    -1.31719995  0.75747001  0.38552001 -0.50441998 -0.55441999
 -0.39649999 -1.13160002  1.22570002  0.22702     0.30836999 -0.18944
  0.49366     0.90425003 -0.45399001 -0.042686   -1.2723     -0.062451
  0.13463999 -0.50247002  0.39923999 -0.36028001 -0.81274998  0.037325
  0.046816   -0.33647001 -1.0474     -0.37382001  0.34393999 -0.50757003
  1.57729995 -0.076262   -0.3581     -0.76959997 -0.19645999  1.02550006
  0.36827001  0.38780999 -0.12588    -0.13531999  0.31990999 -0.03272
 -0.01128     1.47019994 -0.69431001 -0.071377    1.22099996  0.81044
  0.40667999 -0.098573  ]

Unit:   computational
Vector: [  4.31499988e-01  -3.67849991e-02   9.68580022e-02   4.22829986e-01
  -3.88289988e-01   6.89260006e-01   1.01639998e+00  -1.73469996e+00
   1.34930000e-01  -5.69400005e-02   8.11169982e-01   2.79329985e-01
  -6.17060006e-01  -3.97960007e-01  -4.00079995e-01  -2.86139995e-01
   2.48089999e-01   1.27509999e+00   2.92879999e-01  -7.10950017e-01
   8.70049968e-02  -8.45350027e-01  -2.09790006e-01  -2.22760007e-01
   8.37759972e-01   9.81409997e-02  -7.16199994e-01  -8.74830008e-01
  -2.18679994e-01   8.55109990e-01   1.46029997e+00  -7.84169972e-01
  -3.67179990e-01  -1.71550000e+00   9.42170024e-02   8.05830002e-01
  -1.20410001e+00   1.88180006e+00   1.08070004e+00   1.10560000e+00
   4.94690001e-01  -3.08530003e-01  -1.84230000e-01   1.47109997e+00
  -5.90629995e-01  -3.49229991e-01   2.28239989e+00   1.30540001e+00
   1.02009997e-03   1.60899997e-01]

Unit:   computation
Vector: [ 0.44551    -0.20328     0.16670001  0.29977     0.24637     0.44426
  1.08599997 -1.11899996  0.39616001  0.75651002  0.27359    -0.020149
 -0.10735    -0.12139    -0.22418    -0.25176001 -0.028599    0.31507999
  0.25172001 -0.24843     0.22615001 -0.93827999 -0.38602    -0.089497
  0.98723     0.39436001 -0.34908    -0.99075001  0.34147     0.021747
  1.43799996 -0.83107001 -0.48113999 -0.83788002  0.13285001  0.065932
  0.10166     1.00689995  0.10475     0.90570003  0.052845   -0.68559003
 -0.81279999  1.72060001 -1.00870001 -0.61612999  1.9217      0.52373999
  0.0051134   0.23796999]

Unit:   computed
Vector: [ 0.92198998 -0.42993999  1.18130004 -0.60396999  0.58127999 -0.12542
  1.14040005 -1.41620004 -0.091121    0.57312     1.1875      0.33028999
  0.17159     0.20772    -0.23935001  0.91812998 -0.30410999 -0.57440001
 -0.51454002 -0.28658     0.054586   -1.50179994  1.06110001  0.10836
  0.016461    0.57080001 -0.79029    -0.015223   -0.54136997 -0.24146999
  0.77051997  0.14156    -0.038233   -0.84209001  0.10314    -0.41255999
  0.94155002  1.25880003  0.38464999  0.82897002  0.32045999  0.27164999
 -0.77164     1.43519998 -1.39279997 -1.17069995  1.56280005  0.73864001
  0.75353003  0.19359   ]

Unit:   compute
Vector: [ 0.63358003 -0.37999001  1.15170002  0.10287     0.56019002 -0.33078
  0.78088999 -0.52937001  0.36013001  0.049813    0.41021001  0.51063001
  0.023768   -0.73566997 -0.087008    0.44508001  0.23927    -0.13426
  0.53015    -0.84297001 -0.36684999 -1.60409999  0.60742003  0.4862
  0.59741002  0.73307002 -1.10570002 -0.44442001  0.81307    -0.44319999
  1.11520004 -0.14816999 -0.53328001 -0.031922   -0.01878    -0.13345
 -0.0033607   0.33338001  0.41016999  0.45853001  0.56351    -0.59254998
 -0.79004002  1.08350003 -1.11530006 -0.64942002  1.47350001  0.21834999
  0.36024001  0.37728   ]

Unit:   supercomputer
Vector: [ 0.054309   -0.74190003  0.98615003  1.48800004 -0.31690001 -0.79742998
 -0.33346999 -1.24890006  0.48521     0.47497001  0.57542002 -0.14462
 -0.047178    0.71052998 -0.55022001 -0.51172    -0.45679     1.06949997
 -0.86000001 -0.62437999 -0.67954999 -1.68169999 -1.35780001 -0.86707997
  0.23199999 -0.44558001  0.016437   -0.13151    -0.30254    -0.75502998
  0.24353001 -0.51615     0.23749    -0.47378001 -0.86453003 -0.33899
 -0.52517998  1.24790001  0.023642   -0.34333    -0.023264   -0.71818
  0.10802     0.89945     0.62333     0.32117     1.028      -0.053564
 -0.27849001  0.15685   ]

Unit:   supercomputers
Vector: [ 0.13271999 -1.63479996  1.54130006  1.0187     -0.36779001 -0.98526001
  0.18335    -1.27250004  0.43555999  0.35550001  0.38440999  0.059009
  0.093939    0.61080998 -0.026098   -0.25139001 -0.12072     0.90805
 -0.68120003 -1.03770006  0.11673    -1.93009996 -0.45818001 -0.47898
  0.35043001 -0.38150999 -0.14930999 -0.82398999 -0.43788001 -0.30847001
 -0.11093    -0.41409999  0.58244002 -0.18618    -0.065696   -0.18224999
 -0.62984002  1.5941     -0.81909001  0.30436    -0.057413    0.014005
  0.84983999  1.28690004  0.38229001  0.43239999  0.74114001  0.36223999
 -0.61400002 -0.27274001]

Unit:   computations
Vector: [ 0.92869002 -1.02049994  0.19661     0.14015999 -0.11591     0.34413001
  1.30859995 -0.23383    -0.15123001  0.77437001  0.11961     0.14681999
  0.035171    0.23051     0.021644   -0.26311001  0.11231     0.16500001
  0.011065   -0.82683998  0.66431999 -0.88352001 -0.069709   -0.19406
  0.60465002  0.89796001 -0.93678999 -0.94221997  0.026637   -0.65461999
  0.96908998 -0.23707999  0.47549    -0.36783999  0.30926999  0.47736999
  0.75032002  0.92299998 -0.14572001  0.87426001 -0.17066    -0.3971
 -0.38001999  1.71399999 -0.73566997 -0.97488999  1.31379998  0.83398998
 -0.38859999  0.32051   ]

Unit:   computerised
Vector: [ 0.12611    -1.65090001  0.23131999  0.42032    -0.85224003 -0.64967
 -0.10709    -0.82485002  0.82120001  0.013014    0.23706    -0.085659
  0.52227002  0.78956997 -0.73622     0.17614999 -0.94698     0.18522
  0.032076    0.035771    0.20302001 -0.56418997 -0.73012    -0.063655
 -0.079343    0.53434002 -0.23952     0.024863    0.023046   -0.072238
  0.20665    -0.21754    -0.27156001 -0.26984    -0.24496     0.74730998
  0.58513999  0.16144    -0.31505999 -0.11659     0.096848   -0.47889999
 -0.5596      1.82539999 -1.1983      0.10177     0.71583003  0.88134998
  0.63433999 -0.43048999]


<snip> ..... </snip>


Unit:   computec
Vector: [-0.26438001  0.031859    0.37781999  1.19770002  0.037241   -0.28432
 -0.48710001 -0.71013999 -0.097773    1.08249998  0.91813999 -0.11769
  1.06219995  0.95842999 -0.72715002 -0.75755    -1.24370003  0.19340999
  0.74687999 -0.28589001 -1.046       0.21258999 -0.61084998 -0.24936999
  0.45050001  0.79170001 -0.46599001 -0.22724999 -0.72018999  0.24209
 -1.78380001  0.52792001 -0.23574001 -0.35584    -1.83280003 -1.35420001
 -1.56149995 -0.41892999 -0.42469001 -0.65151     0.22994    -0.96930999
  0.25121     0.035985    1.04270005 -0.34784001 -0.34584001 -0.28391001
  0.26899999  0.16615   ]

Unit:   computerise
Vector: [ 0.13243    -1.00460005  0.69104999 -0.46228001 -0.95081002 -0.83868998
  0.50146002  0.96180999  0.66720003 -0.0078055   0.41389999  0.1487
  0.94172001  0.27941    -0.68633997  0.71447998 -0.74552    -0.26036999
  1.26040006  0.12515     0.43461999 -0.22176    -0.1957      0.25902
  0.4844      0.81441998  0.24135999 -0.50159001  0.13429999 -0.31376001
 -1.12609994  0.70595002 -0.18280999  0.14963999 -0.12553     0.17343999
  0.53565001 -0.47918999 -0.73098999 -0.082523    0.13792001 -0.97311002
  0.23997     0.35769999 -0.49739999  0.19893999  0.29245001  0.35404
 -0.33359    -0.29841   ]

Unit:   ncomputing
Vector: [-0.13777    -0.89407998  0.36000001  0.23384    -0.16268    -0.25003001
  0.38916999  0.040075    0.5772      0.38306999  0.17998999  0.11491
  0.47702    -0.16103999 -0.56414002  0.41909999 -0.1071      0.56476998
  0.86243999  0.14602    -0.019593   -0.29097    -0.25075001 -0.075766
  0.14061999  0.73618001  0.24442001  0.25635001 -0.33256     0.32995999
 -1.73239994 -0.65521997  0.42548999 -0.27728999 -0.016066   -0.077929
 -0.44281     0.19193999 -0.24304    -0.42770001  0.15459    -0.18421
 -0.60525    -0.031987    0.054108    0.024123    0.39344999  0.38275999
 -0.40790999  0.47226   ]

Unit:   computrace
Vector: [ 0.032573   -0.20901     0.52177     0.58008999 -0.29374    -0.68484998
  0.39283001  0.24631999  0.91284001  1.19729996 -0.067714    0.14139
  0.20815     0.44073999  0.075302   -0.030624    0.15228     0.12558
  0.86303997  0.24861    -0.41420001 -0.33192    -0.70894998  0.43792
  1.24559999  1.09360003 -0.12145     0.14472     0.64788997 -0.037487
 -0.92712998 -0.21217     0.113       0.61799002 -0.3064      0.19243
 -0.045926    0.10823    -0.13944    -0.33397001  0.10098    -0.45471999
 -0.42684001  0.048138    0.027003    0.40382001  1.00129998  0.26407
  0.51999003  0.084454  ]

Unit:   computacenter
Vector: [ 0.086849   -0.17321     1.00810003  0.21253    -0.5334     -0.13697
  0.56629997  0.68970001  0.47001001  0.65403998 -0.30138999 -0.64124
  0.77232999  0.4826     -0.44688001 -0.12972    -0.034202    0.54593003
  0.41102001  0.45901     0.16802999 -0.65959001 -0.80486     0.30281001
 -0.07883     0.39427999  0.18619999 -0.06051    -0.44953999  1.17190003
 -1.57009995 -0.18610001  0.63310999  0.50357002 -0.20285    -0.48023
 -0.1048      0.41510001 -0.505      -0.89828998  0.14026999 -0.075739
 -0.23270001  0.2129     -0.094783   -0.04949    -0.60021001 -0.24270999
  0.34661001  0.23172   ]
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

However, WOMBAT also provides the ready-to-use standard preprocessor ```wombat_api.preprocessors.standard_preprocessor.py``` (based on NLTK 3.2.5). In order to link it (or <b>any other preprocessing code</b> based on the above stub!!) to one or more WECs in WOMBAT, a pickled instance has to be created first, and then linked to one or more WECs. The following code is available in ```tools/assign_preprocessor_to_glove.py```

</p>

```python
from wombat_api.preprocessors.standard_preprocessor import preprocessor
from wombat_api.core import connector as wb_conn

prepro=preprocessor(name="wombat_standard_preprocessor", phrasefile="")
prepro.pickle("temp/wombat_standard_preprocessor.pkl")

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)
wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token;norm:{none,abtt}", 
                        "temp/wombat_standard_preprocessor.pkl")

# Calling this method with an empty string as pickle file name removes the preprocessor.
# wbc.assign_preprocessor("algo:glove;dataset:6b;dims:{50,100,200,300};fold:1;unit:token;norm:{none,abtt}", "")
```

<p>
After that, raw, unprocessed input data can be streamed directly into WOMBAT's vector retrieval methods.
</p>

```python
import numpy as np
from wombat_api.core import connector as wb_conn

wbpath="data/wombat-data/"
wbc = wb_conn(path=wbpath, create_if_missing=False)

wec_ids="algo:glove;dataset:6b;dims:50;fold:1;unit:token;norm:none"
rawfile="data/text/STS.input.track5.en-en.txt"

vecs = wbc.get_vectors(wec_ids, {}, 
                       for_input=[np.loadtxt(rawfile, dtype=str, delimiter='\t', usecols=0)], 
                       raw=True, 
                       in_order=True, 
                       ignore_oov=True)

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
```
<p>

```ignore_oov=True``` suppresses empty default vectors in the output for oov words (incl. \*sw\* (stop words) produced by the preprocessor).
If the original input ordering need not be preserved (e.g. because vectors of a sentence are averaged anyway), use ```in_order=False``` in order to speed up the retrieval.
Executing this code with

```shell
$ python tools/test_get_vectors_from_raw.py
```

from the WOMBAT directory returns (abbreviated)

</p>

<pre>
WEC: algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
Raw:    'A person is on a baseball team.'
Prepro: ['*sw*', 'person', '*sw*', '*sw*', '*sw*', 'baseball', 'team']
Unit:   person
Vector: [ 0.61734003  0.40035     0.067786   -0.34263     2.06469989  0.60843998
  0.32558     0.38690001  0.36906001  0.16553     0.0065053  -0.075674
  0.57099003  0.17314     1.01419997 -0.49581    -0.38152     0.49254999
 -0.16737001 -0.33948001 -0.44405001  0.77543002  0.20935     0.60070002
  0.86649001 -1.89230001 -0.37900999 -0.28044     0.64213997 -0.23548999
  2.93580008 -0.086004   -0.14327    -0.50160998  0.25290999 -0.065446
  0.60768002  0.13984001  0.018135   -0.34876999  0.039985    0.07943
  0.39318001  1.05620003 -0.23624    -0.41940001 -0.35332    -0.15233999
  0.62158     0.79256999]

Unit:   baseball
Vector: [-1.93270004  1.04209995 -0.78514999  0.91033     0.22711    -0.62158
 -1.64929998  0.07686    -0.58679998  0.058831    0.35628     0.68915999
 -0.50598001  0.70472997  1.26639998 -0.40031001 -0.020687    0.80862999
 -0.90565997 -0.074054   -0.87674999 -0.62910002 -0.12684999  0.11524
 -0.55685002 -1.68260002 -0.26291001  0.22632     0.713      -1.08280003
  2.12310004  0.49869001  0.066711   -0.48225999 -0.17896999  0.47699001
  0.16384     0.16537    -0.11506    -0.15962    -0.94926    -0.42833
 -0.59456998  1.35660005 -0.27506     0.19918001 -0.36008     0.55667001
 -0.70314997  0.17157   ]

Unit:   team
Vector: [-0.62800997  0.12254    -0.39140001  0.87936997  0.28571999 -0.41953
 -1.42649996  0.80462998 -0.27045    -0.82498997  1.02769995  0.18546
 -1.76049995  0.18551999  0.56818998 -0.38554999  0.61609     0.51209003
 -1.51530004 -0.45688999 -1.19289994  0.33886001  0.18038     0.10788
 -0.35567001 -1.57009995 -0.02989    -0.38742    -0.60838002 -0.59188998
  2.99110007  1.20220006 -0.52598    -0.76941001  0.63006002  0.63827997
  0.30772999  1.01230001  0.0050781  -1.03260005 -0.29736    -0.77503997
 -0.27015001 -0.18161     0.04211     0.32168999  0.018298    0.85202003
  0.038442   -0.050767  ]

Raw:    'Our current vehicles will be in museums when everyone has their own aircraft.'
Prepro: ['*sw*', 'current', 'vehicles', '*sw*', '*sw*', '*sw*', 'museums', '*sw*', 'everyone', '*sw*', '*sw*', '*sw*', 'aircraft']
Unit:   current
Vector: [ -9.75340009e-02   7.97389984e-01   4.52930003e-01   8.86869989e-03
  -5.11780009e-02   1.81779992e-02  -1.17909998e-01  -6.97929978e-01
  -1.59400001e-01  -3.38860005e-01   2.13860005e-01   1.19450003e-01
  -3.30779999e-01   7.08459988e-02   5.38580000e-01   5.27660012e-01
  -9.79890004e-02   3.43899988e-02   6.65669963e-02  -2.71719992e-01
   1.15869999e-01  -7.70420015e-01  -2.33769998e-01  -8.57570022e-02
  -2.75379986e-01  -1.26929998e+00   1.56700000e-01  -4.58920002e-02
  -3.45319986e-01   1.30330002e+00   3.62069988e+00   9.13279969e-03
  -1.26800001e-01  -6.15760028e-01   6.60099983e-02  -2.54509985e-01
   1.35349995e-03  -5.12209982e-02  -2.21770003e-01  -4.43280011e-01
  -5.41520000e-01   1.96909994e-01  -3.30339998e-01   3.70520004e-03
  -8.57439995e-01   1.67030007e-01   4.14049998e-02   5.95790029e-01
  -9.78059992e-02   1.86419994e-01]

Unit:   vehicles
Vector: [ 0.75981998 -0.76559001  2.09439993 -0.37478    -0.34946999  0.18489
 -1.11520004 -1.01549995  0.24493    -0.71603     0.60359001 -1.04719996
 -0.28301999 -0.36221999  0.29956001  0.043537   -0.31847     1.47529995
 -0.49761999 -2.1802001   0.52872998 -0.34920001 -0.78740001 -0.058825
 -0.11986    -0.59237999 -0.19368     0.42545     1.21319997  0.19446
  2.66330004  0.30814999 -0.1981     -0.28797999  1.17560005  0.68199998
  0.4655     -0.3504     -1.00339997  0.83025002 -0.2051     -0.24585
  1.10619998 -0.8197      0.26460999 -0.73376    -0.53285003  0.035146
  0.25134    -0.60158002]

Unit:   museums
Vector: [  9.85180020e-01   1.13440001e+00  -6.29760027e-01  -3.34529996e-01
   3.53210010e-02  -1.28009999e+00  -1.04939997e+00  -6.92629993e-01
  -1.51199996e-02  -6.12629987e-02  -1.91709995e-01  -1.35699997e-03
   5.42540014e-01   1.70609996e-01   5.36289990e-01   3.47109996e-02
   8.75020027e-01   4.11379989e-03  -4.10959981e-02   7.34909996e-02
   1.28649998e+00  -2.06609994e-01  -8.32859993e-01   3.66389990e-01
  -6.33740008e-01  -2.20280007e-01  -1.35179996e+00  -3.86290014e-01
  -5.34630001e-01  -1.21969998e+00   1.55239999e+00   6.94739997e-01
   1.02810001e+00  -1.52869999e+00  -5.21550000e-01   8.31290007e-01
   8.52039978e-02   8.92379999e-01  -4.59740013e-01   5.44290006e-01
   1.50869995e-01  -6.45650029e-01   1.70070004e+00   6.50240004e-01
  -1.69949993e-01   9.48629975e-01  -1.07200003e+00   7.92410001e-02
  -5.76539993e-01  -7.30650008e-01]

Unit:   everyone
Vector: [  4.72460017e-02   4.25340012e-02   1.11500002e-01  -5.33339977e-01
   1.14870000e+00  -4.18350011e-01  -4.16669995e-01   4.66320008e-01
  -3.93959992e-02   2.13530004e-01  -1.67190000e-01   2.35850006e-01
  -3.46029997e-01  -3.85849997e-02   1.06449997e+00   4.68389988e-01
   4.45210010e-01   3.39459985e-01   2.97329992e-01  -9.35410023e-01
  -2.72670001e-01   9.17469978e-01  -2.66399998e-02   4.96710002e-01
   1.24520004e+00  -1.83879995e+00  -5.42389989e-01   4.77459997e-01
   9.36029971e-01  -9.21980023e-01   2.71600008e+00   1.13660002e+00
  -2.25899994e-01  -3.84640008e-01  -6.01819992e-01  -2.26870000e-01
   1.16690002e-01   3.29930000e-02   2.30489999e-01  -4.95480001e-01
  -2.52389997e-01   6.36380017e-02  -8.74719992e-02   5.59130013e-01
  -7.14589987e-05   2.49380007e-01  -2.10319996e-01  -2.35870004e-01
  -1.01240002e-01   7.58400023e-01]

Unit:   aircraft
Vector: [ 1.77139997 -0.75713998  1.02170002 -0.26717001 -0.36311001  0.29269001
 -0.79655999 -0.49746001  0.41422001 -1.06019998  1.22150004  0.41672
 -0.40248999  0.70012999 -1.06949997 -0.19489001 -1.08860004  1.24090004
 -2.15050006 -1.1609      0.10969     0.17290001 -0.82805997 -0.97654003
 -0.14616001 -1.26409996 -0.13635001 -0.041624    1.09389997  0.71160001
  2.47399998 -0.16225    -0.26348001  0.15532     1.19949996  0.0076471
  0.76388001 -0.071138   -1.38689995  0.88787001  0.36175001 -0.33419001
  1.65120006 -0.52294999 -0.30656999  0.17399    -0.55383003  0.46204001
 -0.59634     0.41802001]

Raw:    'A woman supervisor is instructing the male workers.'
Prepro: ['*sw*', 'woman', 'supervisor', '*sw*', 'instructing', '*sw*', 'male', 'workers']
Unit:   woman
Vector: [ -1.81529999e-01   6.48270011e-01  -5.82099974e-01  -4.94509995e-01
   1.54149997e+00   1.34500003e+00  -4.33050007e-01   5.80590010e-01
   3.55560005e-01  -2.51839995e-01   2.02539995e-01  -7.16430008e-01
   3.06100011e-01   5.61269999e-01   8.39280009e-01  -3.80849987e-01
  -9.08749998e-01   4.33259994e-01  -1.44360000e-02   2.37250000e-01
  -5.37989974e-01   1.77730000e+00  -6.64329976e-02   6.97950006e-01
   6.92910016e-01  -2.67389989e+00  -7.68050015e-01   3.39289993e-01
   1.96950004e-01  -3.52450013e-01   2.29200006e+00  -2.74109989e-01
  -3.01690012e-01   8.52859986e-04   1.69229999e-01   9.14330035e-02
  -2.36099996e-02   3.62359993e-02   3.44880015e-01  -8.39470029e-01
  -2.51740009e-01   4.21229988e-01   4.86160010e-01   2.23249998e-02
   5.57600021e-01  -8.52230012e-01  -2.30729997e-01  -1.31379998e+00
   4.87639993e-01  -1.04670003e-01]

Unit:   supervisor
Vector: [-0.43483999 -0.29879001 -0.33191001  0.66744    -0.015454   -0.15109
 -0.6063      0.43643999  0.50387001 -1.29209995 -0.19067     0.22946
  0.15900999  0.11937     0.30079001 -0.71973997 -0.76618999  0.40612
  0.45030999 -0.56156999  0.46836001  0.56080002 -0.24398001  0.41773999
 -0.060769   -0.85593998  0.44560999  0.0173     -0.18959001 -0.47902
  1.09940004 -0.39855999 -0.15020999 -1.33490002 -0.23598     0.40862
  0.46061     0.041265    1.44430006  0.25913     0.28817001  0.92123002
 -0.29732999 -0.10582    -0.75729001 -0.40329     0.026871   -0.35651001
  0.38978001  1.96019995]

Unit:   instructing
Vector: [ 0.12468    -0.76235002 -0.036286   -0.89383     0.44255    -0.7999
  0.014672    0.40333    -0.19618    -0.31009001 -0.081948    0.53548002
  0.3971      0.12518001  0.010218   -0.50193    -1.04390001 -0.15561999
  0.9472     -0.46739     0.52798003  0.47464001  0.33513999  0.16192
  0.13628    -0.43952999  0.39326    -0.59561998 -0.43298    -0.79999
  0.30941999  0.40891001 -0.94845003 -0.58431     0.083376    0.27149999
  0.41819    -0.45974001 -0.33594     0.34017     0.31760001 -0.2308
  0.20413999  0.30772999  0.14139999 -0.39932001  0.10814     0.62976003
  0.074504    0.12097   ]

Unit:   male
Vector: [-0.23046     0.65937001 -0.28411001 -0.44365999  1.59220004  1.85640001
 -0.0054708  -0.58679003 -0.1506     -0.021166    1.10290003 -0.79501998
  1.18990004  0.53535002  0.25255999 -0.15882    -0.31825     0.53609002
 -0.59439999 -0.21288    -0.94989002  0.91619003  0.48789999  0.77063
 -0.16215    -1.05149996 -0.70570999 -0.79813999 -0.79354    -0.086372
  2.24970007  0.68785    -0.085613   -0.68004     0.62212002 -0.02536
  0.10967    -0.38747999 -0.62791002 -1.08710003 -0.37412    -0.061965
  0.19225     0.89262998  0.51762998 -1.47909999 -0.23219    -1.15890002
  0.066075   -0.038772  ]

Unit:   workers
Vector: [ 0.47005999 -0.64020002  0.74308002 -0.70699    -0.18398    -0.095573
 -1.12329996  0.66938001  0.31698999 -0.87045002  0.36017999 -1.01370001
  0.60290003 -0.14692     0.65534002 -0.63380003 -0.17293     0.89907002
  0.60336    -1.47580004  0.35749999  0.22641    -0.66198999  0.059413
 -0.36116001 -1.24820006  0.021193   -0.58884001  0.081766    0.16429999
  3.48309994  0.50941998 -0.38088    -0.0052672  -0.38922     0.086958
 -0.047593   -0.56067002  1.07790005  0.53268999 -0.81387001 -0.49265999
  0.92754     0.34024999  0.8642     -0.59026998 -1.4217      0.29286
 -0.31193    -0.34274   ]
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

<h4>Most Similar Words</h4>
<p>

WOMBAT provides the script ```tools/get_most_similar.py``` for computing the most similar words to a given list of target words. The script uses the method ```wombat_api.analyse.get_most_similar```.


```python
import sys
from wombat_api.core import connector as wb_conn
from wombat_api.analyse import get_most_similar
import scipy.spatial.distance as dist

wbpath=sys.argv[1]
wec_ids=sys.argv[2]
targets=sys.argv[3].split(",")
try:
    to_rank=sys.argv[4].split(",")
except IndexError:
    to_rank=[]

wbc = wb_conn(path=wbpath, create_if_missing=False)

sims = get_most_similar(wbc, wec_ids, targets=targets, measures=[dist.cosine], to_rank=to_rank)
for (w, wec, mes, simlist) in sims:
    print("\n%s"%(wec))
    for (t,s) in simlist:
        print("%s(%s, %s)\t%s"%(mes,w,t,s))

```
Computing the similarity of a given list of target words to *all* words in an embedding set is a task that does not benefit from Wombat's lazy loading philosophy, because it involves iterating over a lot of single items. The above code compensates this by accepting several target words at once, while loading the words in the embedding set only once.

Executing the script with

```shell
$ python tools/get_most_similar.py "data/wombat-data/" "algo:glove;dataset:6b;dims:{50,100};fold:1;norm:{none,abtt};unit:token" car,bike
```
from the WOMBAT directory returns 

<pre>
algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
cosine(car, truck)	0.07914144136184864
cosine(car, cars)	0.11298109069525497
cosine(car, vehicle)	0.11663159684321234
cosine(car, driver)	0.15359811852812422
cosine(car, driving)	0.16158120657580843
cosine(car, bus)	0.17894889497726807
cosine(car, vehicles)	0.18250077858745317
cosine(car, parked)	0.2097811084657102
cosine(car, motorcycle)	0.2133497199448282
cosine(car, taxi)	0.21660710099093428

algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
cosine(bike, bicycle)	0.07540862422613559
cosine(bike, rides)	0.12897087378541827
cosine(bike, bikes)	0.15252882825561032
cosine(bike, ride)	0.16029085596645365
cosine(bike, cart)	0.20388619664671093
cosine(bike, bicycles)	0.22393171208065155
cosine(bike, riding)	0.2297407298062787
cosine(bike, motorcycle)	0.24199681247288152
cosine(bike, skateboard)	0.24562024322931186
cosine(bike, wheel)	0.24976224925775947

algo:glove;dataset:6b;dims:50;fold:1;norm:abtt;unit:token
cosine(car, truck)	0.0806001419007456
cosine(car, driver)	0.12179994387193638
cosine(car, vehicle)	0.1385399783711604
cosine(car, cars)	0.14205120673399707
cosine(car, tractor)	0.19330317428597177
cosine(car, cab)	0.19371578595889627
cosine(car, driving)	0.1967477518121835
cosine(car, taxi)	0.19764512986360383
cosine(car, parked)	0.2024978715831982
cosine(car, forklift)	0.21243824560524704

algo:glove;dataset:6b;dims:50;fold:1;norm:abtt;unit:token
cosine(bike, bicycle)	0.08398014976833035
cosine(bike, rides)	0.1430640377058503
cosine(bike, bikes)	0.16369354577451944
cosine(bike, ride)	0.17653528980791744
cosine(bike, limo)	0.1823194282582885
cosine(bike, skateboard)	0.2085667400501673
cosine(bike, cart)	0.21514646350843625
cosine(bike, bicycles)	0.23932357247389668
cosine(bike, riding)	0.25687287619295995
cosine(bike, biking)	0.26260029724823075

algo:glove;dataset:6b;dims:100;fold:1;norm:none;unit:token
cosine(car, vehicle)	0.13691616910455218
cosine(car, truck)	0.1402122094746816
cosine(car, cars)	0.16283305313114194
cosine(car, driver)	0.18140894723421486
cosine(car, driving)	0.21873640792744087
cosine(car, motorcycle)	0.2446842503669403
cosine(car, vehicles)	0.25377434558164547
cosine(car, parked)	0.2540535380120613
cosine(car, bus)	0.26272929599923434
cosine(car, taxi)	0.28447302367774396

algo:glove;dataset:6b;dims:100;fold:1;norm:none;unit:token
cosine(bike, bicycle)	0.10315127761665555
cosine(bike, bikes)	0.20443421876273637
cosine(bike, ride)	0.22046929133315563
cosine(bike, rides)	0.2638311426114084
cosine(bike, riding)	0.27133477109461057
cosine(bike, motorcycle)	0.27805119727347305
cosine(bike, biking)	0.2816471833865629
cosine(bike, horseback)	0.31557397925187236
cosine(bike, bicycles)	0.3187722929261676
cosine(bike, riders)	0.3254949790131334

algo:glove;dataset:6b;dims:100;fold:1;norm:abtt;unit:token
cosine(car, truck)	0.15238329488374347
cosine(car, vehicle)	0.15575847257407438
cosine(car, cars)	0.19167657709380725
cosine(car, driver)	0.20033349172277293
cosine(car, parked)	0.24794750003421806
cosine(car, motorcycle)	0.2510652900482522
cosine(car, driving)	0.25658421356403294
cosine(car, suv)	0.2881546903629949
cosine(car, bus)	0.2910614135644427
cosine(car, vehicles)	0.29615907557187104

algo:glove;dataset:6b;dims:100;fold:1;norm:abtt;unit:token
cosine(bike, bicycle)	0.1088470577560825
cosine(bike, bikes)	0.21590419939848782
cosine(bike, ride)	0.23369856648438625
cosine(bike, rides)	0.27806636584727484
cosine(bike, biking)	0.2832740671069537
cosine(bike, riding)	0.28638550538216256
cosine(bike, motorcycle)	0.2913097546696938
cosine(bike, horseback)	0.324846874936749
cosine(bike, bicycles)	0.3404461149572644
cosine(bike, wagon)	0.3443322594384779

</pre>
The above code takes some time, though.

</p>

<p>
Things are a lot different when only a small list of words is to be ranked according to their similarity to one or more target words.
Executing the above script with an additional list of words like this

```shell
$ python tools/get_most_similar.py "data/wombat-data/" "algo:glove;dataset:6b;dims:{50,100};fold:1;norm:{none,abtt};unit:token" car,bike trolley,bus,vehicle,transporter
```
from the WOMBAT directory returns 

<pre>
algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
cosine(car, vehicle)	0.11663159684321234
cosine(car, bus)	0.17894889497726807
cosine(car, trolley)	0.48697765622473255
cosine(car, transporter)	0.6139896275893459

algo:glove;dataset:6b;dims:50;fold:1;norm:none;unit:token
cosine(bike, vehicle)	0.3427957759292295
cosine(bike, bus)	0.34365947338677905
cosine(bike, trolley)	0.3602480028404018
cosine(bike, transporter)	0.7320497642797394

algo:glove;dataset:6b;dims:50;fold:1;norm:abtt;unit:token
cosine(car, vehicle)	0.1385399783711604
cosine(car, bus)	0.2158960678290227
cosine(car, trolley)	0.46696018041448584
cosine(car, transporter)	0.5406758968293157

algo:glove;dataset:6b;dims:50;fold:1;norm:abtt;unit:token
cosine(bike, trolley)	0.3678464886357319
cosine(bike, vehicle)	0.3874397902633365
cosine(bike, bus)	0.3921970555479769
cosine(bike, transporter)	0.7319556230922035

algo:glove;dataset:6b;dims:100;fold:1;norm:none;unit:token
cosine(car, vehicle)	0.13691616910455218
cosine(car, bus)	0.26272929599923434
cosine(car, trolley)	0.5475087400049348
cosine(car, transporter)	0.7290820977867609

algo:glove;dataset:6b;dims:100;fold:1;norm:none;unit:token
cosine(bike, trolley)	0.38364037699224673
cosine(bike, bus)	0.44165326460377197
cosine(bike, vehicle)	0.4536933011117086
cosine(bike, transporter)	0.8071001886680546

algo:glove;dataset:6b;dims:100;fold:1;norm:abtt;unit:token
cosine(car, vehicle)	0.15575847257407438
cosine(car, bus)	0.2910614135644427
cosine(car, trolley)	0.5404368768171397
cosine(car, transporter)	0.6956990227076467

algo:glove;dataset:6b;dims:100;fold:1;norm:abtt;unit:token
cosine(bike, trolley)	0.3900553987623596
cosine(bike, bus)	0.4667747849371262
cosine(bike, vehicle)	0.48185728456605526
cosine(bike, transporter)	0.807988795692304

</pre>

</p>

