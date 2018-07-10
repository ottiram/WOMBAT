import sys, pickle
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from tqdm import tqdm
from wombat_api.lib import rawcount, PROGBARWIDTH

def pre_tokenized(doc):
    """doc is a list of tokenized lists (with pre-tokenized values)
    will be passed to sklearn to bypass analyzer"""
    return doc

def compute_idf_lookup(collector, outfile):
    tf = TfidfVectorizer(analyzer=pre_tokenized)
    try:
        tf.fit_transform(collector.values())
        vocab = tf.vocabulary_
        idf = tf.idf_             
    except Exception as ex:
        print(ex)
        vocab = {}
    out_file = open(outfile,"w")
    for w, i in vocab.items():
        out_file.write(w+"\t"+str(idf[i])+"\n")
    out_file.close()
    return 


if __name__ == "__main__":  

    infilename=sys.argv[1]
    preproname=sys.argv[2]

    # stopword strategies:
    # sw_symbol=None  : No special stopword handling
    # sw_symbol=""    : Completely remove stopwords
    # sw_symbol="SYM" : Replace all stopwords by SYM
    # conflate=True   : (Only if sw_symbol="SYM": Conflate sequences of SYM to single SYM

    conflate_sws    = False
    conflabel       = ""
    fold            = False
    sw_symbol       = None
    unit            = ""
    repeat_phrases  = False
    repeat_phrases_label = ""

    for a in sys.argv:
        if a.lower() == "conflate":
            conflate_sws = True
            conflabel=".conflated_sws"
        elif a.lower().startswith("stopwords:"):
            sw_symbol = a.lower()[10:]
        elif a.lower().startswith("unit:"):
            unit = a.lower()[5:]
        elif a.lower()=="repeat_phrases":
            repeat_phrases=True
            repeat_phrases_label=".repeat_phrases"
        elif a.lower()=="fold":
            fold=True


    if (sw_symbol=="None" or sw_symbol=="") and conflate_sws==True:
        print("Conflate works with explicit sw symbol (stopwords:XXX) only!")
        sys.exit()

    if unit=="":
        print("Missing unit!")
        sys.exit()

    prepro = pickle.load(open(preproname,"rb"), encoding="utf-8")

    lines_to_process=0
    try:
        lines_to_process=rawcount(infilename)
    except Exception: raise 

    # Create actual output
    with open(infilename, 'r') as infile,\
        open(infilename+conflabel+repeat_phrases_label+"."+unit, 'w') as max_unitfile,\
        open(infilename+conflabel+".nophrases."+unit, 'w') as simple_unitfile :

        simple_unit_bg, max_unit_bg={},{}

        bar = tqdm(total=lines_to_process, ncols=PROGBARWIDTH)
        for line in infile:

            max_units       = prepro.process(line, unit, fold=fold, sw_symbol=sw_symbol, conflate=conflate_sws)
            simple_units    = prepro.process(line, unit, fold=fold, sw_symbol=sw_symbol, conflate=conflate_sws, no_phrases=True)

            # Create embedding training data lines
            for t in simple_units:
                simple_unitfile.write(t+" ")
            simple_unitfile.write("\n")
            # Update count for idf for simple units
            simple_unit_bg[str(len(simple_unit_bg))] = simple_units

            for t in max_units:
                max_unitfile.write(t+" ")
            max_unitfile.write("\n")
            # Update count for idf for max units
            max_unit_bg[str(len(max_unit_bg))] = max_units

            if repeat_phrases:                
                # Add extra emb training line with split phrases, if any
                if len(simple_units) > len(max_units):
                    phrase_parts=[]
                    for b in max_units:
                        if b.find("_") >-1:
                            phrase_parts.extend(list(b.split("_")))
                    if len(phrase_parts)>0:
                        # There was at least one phrase in this line, so add *full **split** line* to emb training data *again*
                        for t in simple_units:
                            max_unitfile.write(t+" ")
                        max_unitfile.write("\n")
                        # Update count for idf for phrase *parts* only
                        max_unit_bg[str(len(max_unit_bg))] = phrase_parts
            bar.update(1)
        bar.close()                                 
        compute_idf_lookup(max_unit_bg,infilename+conflabel+repeat_phrases_label+"."+unit+".idf")
        compute_idf_lookup(simple_unit_bg,infilename+conflabel+".nophrases."+unit+".idf")


class SplitSentence(object):
    def __init__(self, filename):
        self.filename = filename
 
    def __iter__(self):
        for line in open(self.filename):
            yield line.split()


