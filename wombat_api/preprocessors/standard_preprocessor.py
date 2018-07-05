from nltk.tokenize.moses import MosesTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import re, html, pickle
from wombat_api.phrasespotter import phrasespotter

SW_SYMBOL="*sw*"

class preprocessor(object):
    def __init__(self, name=__name__, phrasefile="", verbose=False):
        if verbose: print("Initializing preprocessor %s"%name)
        self.TOKENIZER = MosesTokenizer(lang='en')
        self.STEMMER = PorterStemmer(mode='NLTK_EXTENSIONS')
        self.STOPWORDS = set(stopwords.words('english'))
        self.TAGS_RE = re.compile('<.*?>')                        
        self.PHRASESPOTTER = None if phrasefile=="" else phrasespotter(phrasefile=phrasefile, verbose=verbose)


    """ This method is called from WOMBAT.
        sw_symbol == None means no replacement, use empty string for sw removal. """
    def process(self, line, unit, fold=True, sw_symbol=SW_SYMBOL, conflate=False, no_phrases=False, verbose=False): 

        # Do some global things here on the whole line
        # This replaces agressive_hypen_split
        line = line.strip().replace("-"," - ")

        # R&D etc.
        line = re.sub("r\s?&amp;\s?d"," r-and-d ",line)
        line = re.sub("R\s?&amp;\s?D"," R-and-D ",line)

        # 2-d etc.
        line = re.sub("2\s?-?\s?d","2d",line)
        line = re.sub("2\s?-?\s?D","2D",line)
        line = re.sub("3\s?-?\s?d","3d",line)
        line = re.sub("3\s?-?\s?D","3D",line)

        # Lowercase only if fold==True
        if fold: line=line.lower()
               
        line = line.strip().replace("--"," -- ")
        line = re.sub(self.TAGS_RE,'', html.unescape(line))

        if no_phrases==True:
            if unit     =="token": return self.tokenize(line, sw_symbol=sw_symbol, conflate=conflate)
            elif unit   == "stem": return self.stem(self.tokenize(line,sw_symbol=sw_symbol, conflate=conflate))
        elif self.PHRASESPOTTER != None:
            if unit     =="token": return self.PHRASESPOTTER.extract(self.tokenize(line, sw_symbol=sw_symbol, conflate=conflate))
            elif unit   == "stem": return self.PHRASESPOTTER.extract(self.stem(self.tokenize(line,sw_symbol=sw_symbol, conflate=conflate)))
        else:
            #print("No phrases for this preprocessor, ignoring input phrases.")
            if unit     =="token": return self.tokenize(line, sw_symbol=sw_symbol, conflate=conflate)
            elif unit   == "stem": return self.stem(self.tokenize(line,sw_symbol=sw_symbol, conflate=conflate))
     


    def tokenize(self, line, sw_symbol=None, conflate=False):
        temp_tokens = self.TOKENIZER.tokenize(line, agressive_dash_splits=False, escape=False)
        tokens=[]
        for t in temp_tokens:
            # Get each raw token
            t = t.strip()
            # With each token
            while(True):            
                modified=False
                if t.isnumeric(): 
                    t="0"*len(t)
                    break   # No more mods apply, break here

                if len(t)==1 and is_plain_alpha(t)==False:
                    t=""                    
                    break   # No more mods apply, break here                

                if t.lower() in self.STOPWORDS and sw_symbol!=None:
                    # t is a stopword, which is to be treated somehow
                    if conflate:
                        # stopword conflation is active
                        try:
                            # If the last token is a sw already, skip current sw token
                            if tokens[-1]==sw_symbol:
                                t=""
                                break
                            else:
                                t=sw_symbol
                                break
                        except IndexError:
                            # current sw is first token
                            t=sw_symbol
                            break
                    else:
                        t=sw_symbol # sw-symbol might also be empty, in which case an empty string is added here, which is removed later
                        break   # No more mods apply, break here

                if t!="" and t not in {'2d', '2D', '3d', '3D', "'s"}:
                    if is_plain_alpha(t[0])==False:
                        t=t[1:].strip()
                        modified=True
                    elif is_plain_alpha(t[-1])==False:
                        t=t[:-1].strip()
                        modified=True                        
                if modified==False: # Do this as last point in check
                    break
            if t!="": tokens.append(t)
        return tokens

    def stem(self, tokens):
        stems=[]
        for t in tokens:
            stems.append(self.STEMMER.stem(t))
        return stems    

    def pickle(self, picklefile):
        pickle.dump(self, open(picklefile,"wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Written to %s"%picklefile)


def is_plain_alpha(t):
    o=ord(t)
    if  (o>= 65 and  o<=90) or\
        (o>= 97 and o<=122) or\
        (o>=192 and o<=214) or\
        (o>=216 and o<=246) or\
        (o>=248 and o<=476) or\
        (o>=512 and o<=591):
        return True
    else:
        return False

