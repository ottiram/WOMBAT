import sys
from wombat_api.core import connector as wb_conn

wbpath=sys.argv[1]
wec_ids=sys.argv[2]
pattern=sys.argv[3]
try:                exclude_pattern=sys.argv[4]
except IndexError:  exclude_pattern=""

wbc = wb_conn(path=wbpath, create_if_missing=False)
vecs = wbc.get_matching_vectors(wec_ids, pattern=pattern, exclude_pattern=exclude_pattern)

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

