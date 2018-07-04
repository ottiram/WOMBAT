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
The following code accesses two of the four GloVe WECs and looks up &lt;unit, vector&gt; tuples for two sequences of words.
</p>

<pre>
from wombat_api.core import connector as wb_conn
wbpath="data/wombat-data/"
wec_ids="algo:glove;dataset:6b;dims:{50,100};fold:1;unit:token"
wbc = wb_conn(path=wbpath, create_if_missing=False)

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
