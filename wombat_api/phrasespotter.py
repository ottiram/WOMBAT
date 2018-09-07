class phrasespotter(object):

    def __init__(self, phrasefile="", verbose=False):
        self.PHRASES=set()
        for i in open(phrasefile):
            i=i.strip()
            if i.find(" ")>0:
                self.PHRASES.add(i)
            else: print("Skipping single-token word "+i)
        print("Read %s phrases"%str(len(self.PHRASES)))
        if verbose: print(self.PHRASES)
        return

    """
    This method is called to apply the finished phrasespotter to a list of tokens
    """
    def extract(self, tokens, max_n=0, verbose=False):
        # This will always find the *maximal* phrase, and no embedded ones.
        if verbose: print(tokens)
        # If max_n is not set, use entire input list
        limit_to=len(tokens) if max_n==0 else max_n
        # Move through tokens from left to right
        for o in range(len(tokens)):
            # Move through tokens from right to left, start at limit_to
            for n in range(limit_to,1,-1):
                ngram=" ".join(tokens[o:o+n])
                if verbose: print(ngram)
                if ngram in self.PHRASES:
                    # Join tokens from o to n as a phrase, continue recursively
                    if verbose: print("Found phrase '%s'"%ngram)
                    return self.extract(tokens[:o] + [ngram.replace(" ","_")] + tokens[o+n:], max_n=max_n, verbose=verbose)
        return tokens

