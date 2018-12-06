import itertools, io, re, sys, sqlite3, copy
import numpy as np
import scipy.spatial.distance as dist
from sklearn.decomposition import PCA


MASTER_DB_NAME  =   "wombat_master.sqlite"
PROGBARWIDTH    =   70
CHUNKSIZE       =   300

NAMED_DISTANCE_MEASURES = {'cosine':dist.cosine, 'euclidean':dist.euclidean, 'cityblock':dist.cityblock, 'canberra':dist.canberra, 'chebyshev':dist.chebyshev}

def expand_parameter_grids(we_params, base_figsize=(5,5)):
    """ Receives a string of the form "att1:val;att2:{vala,valb};att3:{,valc}"
        and returns a list of dictionaries resulting from the expansion of the alternative values,
        where attributes are expanded in their supplied order.
        Also returns a list containing the sizes of the sets of alternative values, used for plotting.
    """
    temp_list, param_dict_list, grid_sizes, plot_coords = [],[],[],[]

    if type(we_params) is dict: we_params = dict_to_sorted_string(we_params, pretty=True)
    for we_param in we_params.split("&"):
        temp_list=[]
        for p in we_param.strip().split(";"):
            attribute = p.split(":")[0]
            values = p.split(":")[1].split(",")
            if len(values)>1:
                values[0]=values[0][1:].strip()                     # Cut { from first att value
                values[-1]=values[-1][:len(values[-1])-1].strip()   # Cut } from last att value
                grid_sizes.append(len(values))
            # Using a list here instead of a dict allows to maintain the supplied order, which is used to
            # control the layout of plotted grid search results. 
            temp_list.append((attribute, values))
        keys, values = zip(*temp_list)
        dl = [dict(zip(keys, v)) for v in itertools.product(*values) if v]
        for d in dl: param_dict_list.append(dict((k, v) for k, v in iter(d.items()) if v))

        # Expand conditional values of the form [dim==100->1|dim==200->2|dim==300->3]
        for d in param_dict_list:
            for k,v in d.items():
                if v.startswith("["):
                    rules = (v[1:-1]).split("|")
                    for r in rules:
                        att=r.split("==")[0]
                        if att in d.keys():    
                            att_val=r.split("==")[1].split("->")[0]
                            if d[att] == att_val:
                                d[k] = r.split("==")[1].split("->")[1]

        # Remove all conditional values that were not expanded (i.e. that do not apply)
        to_delete=[]
        for d in param_dict_list:
            for k,v in d.items():
                if v.startswith("["): to_delete.append(k)    
        for k in to_delete:
            del d[k]


        # param_dict_list contains one dict for each expansion result in we_param_string, in the order in which the results to expand were supplied in we_param_string
        # grid_sizes contains, in the same order, the number of supplied different values for each multi-value attribute in we_param_string (up to three)
    
        # Make sure grid_sizes contains exactly three items, for row, col, and page rendering
        if   len(grid_sizes)==0:    grid_sizes=[1,1,1]          # 1 row, 1 col, 1 page, No grids, will result in one plot only 
        elif len(grid_sizes)==1:    grid_sizes.extend([1,1])    # add dummy col and page, One grid, will produce several rows
        elif len(grid_sizes)==2:    grid_sizes.append(1)        # add dummy page, Two grids, will produce several rows and cols in one page
        elif len(grid_sizes)>3 :
            #print("More than three dims in grid not supported for plotting!")
            pass

        # Store x-y coords for all plots on all pages
        for row in range(grid_sizes[0]):
            for col in range(grid_sizes[1]):
                for page in range(grid_sizes[2]):
                    plot_coords.append((row,col,page))
    
        rows=grid_sizes[0]
        cols=grid_sizes[1]
        pages=grid_sizes[2]

    return (param_dict_list, plot_coords, rows, cols, pages)


def abtt_postproc(V, dtype=np.float32, D=0):
    print(dtype)
    assert D >= 1
    V_ = V - V.mean(0)
    pca = PCA()
    pca.fit(V_)
    s = np.zeros(V.shape, dtype=dtype)
    for i in range(D):
        s += np.repeat((pca.components_[i] * V_).sum(1)[:, None], V.shape[1], 1) * pca.components_[i]
    return V_ - s


"""
Utility methods
"""

""" 
For inserting np objects into sqlite
"""
def input_array_adapter(arr):
    """
    http://stackoverflow.com/a/31312102/190597 (SoulNibbler)
    """
    out = io.BytesIO()
    np.save(out, arr)
    out.seek(0)
    return sqlite3.Binary(out.read())

""" 
For selecting np objects from sqlite
"""
def output_array_converter(blob):
    out = io.BytesIO(blob)
    out.seek(0)
    return np.load(out)

def dict_to_sorted_string(desc_dict, pretty=False):
    """ Receives a dictionary with key:val pairs and returns a stringified list, sorted by key.   
        Used to create a unique key string from a dictionary of embedding parameters.   
    """ 
    templist = []
    for  k,v in iter(desc_dict.items()):
        templist.append(k+":"+v)
    if pretty:
        t = str(sorted(templist)).replace("['","").replace("']","").replace("', '",";")
        return t
    else:
        return str(sorted(templist))



def query_yes_no(question, default="no"):
    #  http://code.activestate.com/recipes/577058/
    """Ask a yes/no question via input() and return its answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes", "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "yo": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("Invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")            

def rawcount(filename):
    f = open(filename, 'rb')
    lines = 0
    buf_size = 1024 * 1024
    read_f = f.raw.read

    buf = read_f(buf_size)
    while buf:
        lines += buf.count(b'\n')
        buf = read_f(buf_size)
    return lines


""" 
Console output coloring
"""
class C:
   PURPLE       = '\033[95m'
   CYAN         = '\033[96m'
   DARKCYAN     = '\033[36m'
   BLUE         = '\033[94m'
   GREEN        = '\033[92m'
   YELLOW       = '\033[93m'
   RED          = '\033[91m'
   BOLD         = '\033[1m'
   UNDERLINE    = '\033[4m'
   END          = '\033[0m'



# Wombat Exceptions
class NoSuchWombatEmbeddingsException(Exception): 
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class NoSuchWombatMasterDBException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class WombatEmbeddingDatabaseExistsException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)

class MissingRequiredPropertyException(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return str(self.value)


