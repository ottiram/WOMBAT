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

