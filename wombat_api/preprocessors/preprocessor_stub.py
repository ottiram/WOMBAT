import pickle

# Stop-word replacement
SW_SYMBOL="*sw*"

class preprocessor(object):
    def __init__(self, name=__name__, phrasefile="", verbose=False):

        if verbose: print("Initializing preprocessor %s"%name)

    """ This method is called from WOMBAT.
        'line' is the raw string to be processed,
        'unit' is the processing unit to be used. 
    """
    def process(self, line, unit, fold=True, sw_symbol=SW_SYMBOL, conflate=False, no_phrases=False, verbose=False): 

        # Lowercase if fold==True
        if fold: line=line.lower()
        # This does the most rudimentary preprocessing only
        return line.split(" ")        

    def pickle(self, picklefile):
        pickle.dump(self, open(picklefile,"wb"), protocol=pickle.HIGHEST_PROTOCOL)
        print("Written to %s"%picklefile)

