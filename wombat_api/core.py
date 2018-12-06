import sys, sqlite3, os, itertools, io, re, pickle, base64
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
from wombat_api.lib import *

sqlite3.register_adapter(np.ndarray, input_array_adapter)
sqlite3.register_converter("BLOB", output_array_converter)

"""
The connector class is the single entry point for all emb db accesses.
"""
class connector(object):
    def __init__(self, path="", create_if_missing=False, list_contents=False):
        self.WM     = None
        self.PATH   = ""
        version     = "2.1"
        # This raises and passes on an exception if the masterdb does not exist, and we do not want to create one
        try: self.WM=masterdb(path=path, create_if_missing=create_if_missing)
        except NoSuchWombatMasterDBException as ex: raise
        self.PATH = path 
        (db_count,) = self.WM.DB.cursor().execute("select count(*) from we_meta").fetchone()
        print("\n\n"+C.RED+C.BOLD+" |                                   | ")
        print(" |             ,.--\"\"\"\"--.._         |")
        print(" |           .\"     .'      `-.      |")
        print(" |          ;      ;           ;     |")
        print(" |         '      ;             )    |")
        print(" |        /     '             . ;    |")
        print(" |       /     ;     `.        `;    |")
        print(" |     ,.'     :         .    : )    |")
        print(" |     ;|\\'    :     `./|) \  ;/     |")
        print(" |     ;| \\\"  -,-  \"-./ |;  ).;      |")
        print(" |     /\/             \/   );       |")
        print(" |    :                \     ;       |")
        print(" |    :     _      _     ;   )       |")
        print(" |    `.   \;\    /;/    ;  /        |")
        print(" |      !    :   :     ,/  ;         |")
        print(" |       (`. : _ : ,/""     ;          |")
        print(" |        \\\\\`\"^\" ` :    ;           |"+C.GREEN+"   This is "+\
        C.BOLD+ "WOMBAT"+C.END+C.GREEN+", the "+\
        C.BOLD+"WO"+C.END+C.GREEN+"rd e"+C.BOLD+"MB"+C.END+C.GREEN+"edding d"+C.BOLD+"AT"+C.END+C.GREEN+"a base"+\
        C.BOLD+" API (Version "+version+")"+C.END+C.RED)
        print(C.BOLD+" |                (    )             |"+ C.END+\
        "   Connected to WOMBAT Master DB "+C.BLUE+C.BOLD+os.path.dirname(path)+"/"+MASTER_DB_NAME+C.END)
        print(C.RED+C.BOLD+" |                 ////              |   "+C.END+"with "+C.BLUE+C.BOLD+str(db_count)+\
        C.END+" word embedding sets.")
        print(C.RED+C.BOLD+" |                                   |"+C.END)
        print(C.RED+C.BOLD+" | Wombat artwork by akg             |"+C.END)
        print(C.RED+C.BOLD+" |            http://wombat.ascii.uk |"+C.END)
        print(C.END)
        if list_contents:
            self.contents()
    # end of init()


    """
    Assign preprocessor to the embedding db(s) specified by we_param_grid_string.
    The code is provided as a pickled instance.
    wec_identifier can use the form with curly braces: algo:glove;dataset:6b;dims:{50,100};fold:1;unit:token
    """
    def assign_preprocessor(self, wec_identifier, prepro_picklefile=""):
        we_param_dict_list,_,_,_,_ = expand_parameter_grids(wec_identifier)
        prepro_file_name=os.path.basename(prepro_picklefile)
        if prepro_picklefile !="":
            with open(prepro_picklefile, 'rb') as f:
                # No need to unpickle here ...
                prepro_bin_string = base64.encodestring(f.read())
            message="Trying to assign "+prepro_file_name+" to "
        else:            
            prepro_bin_string=""
            message="Trying to remove prepro_code from "
        
        for we_param_dict in we_param_dict_list:
            desc=dict_to_sorted_string(we_param_dict)
            print(message+desc)
            #print("Assigning %s to %s ... "%(prepro_file_name, desc))
            # Check for existence
            (exists,)= self.WM.DB.cursor().execute("select count(*) from we_meta where descriptor = ?",(desc, ))
            if exists[0] == 1:
                self.WM.DB.cursor().execute("update we_meta set prepro_name = ?, prepro_code = ? where descriptor = ?",(prepro_file_name, prepro_bin_string, desc))
                self.WM.DB.cursor().execute("commit")
                print("\tDone!")
            else:
                print("\tNot found, skipping!")
    # end assign_prepro_code


    """ 
    Plain streaming import for large files. Will be more memory-intensive if normalize is used.
    we_param_grid_string can use the list form, but must specify exactly one wec: algo:glove;dataset:6b;dims:50;fold:1;unit:token
    """
    def import_from_file(self, import_file_name, wec_identifier, dtype=np.float32, prepro_picklefile="", normalize="none"):
        we_param_dict_list,_,_,_,_ = expand_parameter_grids(wec_identifier)
        if len(we_param_dict_list) > 1:
            print("import_from_file() supports single file import only.")
            sys.exit()
        we_param_dict=we_param_dict_list[0]
        desc=dict_to_sorted_string(we_param_dict)
        print("Trying to import DB for descriptor %s"%desc)

        # Check if required db exists
        db_meta = self.WM._get_embeddingdb_meta(desc)
        if len(db_meta)>0:
            print(C.RED+C.BOLD+"Embedding DB already exists for descriptor\n"+\
                desc+C.END+"\n"+C.YELLOW+"-->"+db_meta['db_path']+C.END)
            print("Updating DBs is not supported. DB will not be modified!\n")
            sys.exit()

        lines_to_read=0
        print("Scanning number of lines to import ...")
        try:
            lines_to_read=rawcount(import_file_name)
            in_file = open(import_file_name,"r")
        except Exception: raise 

        emb_db = self.WM._get_embdb(we_param_dict, create=True)
        emb_db.DB.cursor().execute('PRAGMA temp_store = 2;')
        normalize=normalize.lower()
        if normalize=="none":
            print("Doing direct DB insert ... ")
            b = tqdm(total=lines_to_read, ncols=PROGBARWIDTH)
            # Do single insert to track constraint violations
            for e in in_file:
                if len(e.split(" "))<=2: continue
                pos = e.find(" ")
                word = e[0:pos].strip()
                vector = np.fromstring(e[pos+1:].strip(), dtype=dtype, sep=" ")
                try: emb_db.DB.cursor().execute('INSERT INTO vectors (word, vector) values (?,?)', (word,vector))
                except sqlite3.IntegrityError as ex:
                    print(str(ex)+" :"+word)
                    continue
                b.update(1)
            b.close()
        else:
            print("Normalizing with %s. Reading file ... "%normalize)
            b = tqdm(total=lines_to_read, ncols=PROGBARWIDTH)
            vectors=[]
            words=[]
            for e in in_file:
                if len(e.split(" "))<=2: continue
                pos = e.find(" ")
                words.append(e[0:pos].strip())
                vectors.append(np.fromstring(e[pos+1:].strip(), dtype=dtype, sep=" "))
                b.update(1)
            print("\nNormalizing ...")
            b = tqdm(total=lines_to_read, ncols=PROGBARWIDTH)
            if normalize in ('l1','l2','max'):
                n=preprocessing.normalize(vectors,norm=normalize,axis=1, dtype=dtype)#,copy=True,return_norm=True)
            elif normalize == "abtt":
                dim=int(we_param_dict['dims']) # get dim for D
                n=abtt_postproc(np.vstack(vectors),D=max(1,int(dim/100)), dtype=dtype)
            for word, vector in zip(words,n):
                try: emb_db.DB.cursor().execute('INSERT INTO vectors (word, vector) values (?,?)', (word,vector))
                except sqlite3.IntegrityError as ex:
                    print(str(ex)+" :"+word)
                    continue
                b.update(1)            
            b.close()    
        emb_db.DB.commit()
        in_file.close()
                
        if prepro_picklefile != "":
            assign_preprocessor(wec_identifier, prepro_picklefile)
    # end import_from file


    def get_all_vectors(self, wec_identifier, as_tuple=True, verbose=False):
        (we_params_dict_list, _, _, _, _)=expand_parameter_grids(wec_identifier)
        total_result = []

        for we_param_dict in we_params_dict_list:
            we_id = dict_to_sorted_string(we_param_dict, pretty=True)
            if verbose: print(we_id)
            result_for_we = []
            try:
                embdb=self.WM._get_embdb(we_param_dict)
            except NoSuchWombatEmbeddingsException as ex:
                print(ex)
                continue
            result_for_we.append(('',[],embdb.get_all_vectors(as_tuple=as_tuple)))

            total_result.append((we_id, result_for_we))
        return total_result


    """
    Get vectors for input of various types from one or more we databases.
    'for_input' is a list of lists of either *raw strings* or *preprocessed tokens* : 
    1. [['This is a raw string!'], ['Another one!']],                               (e.g. for sentence similarity)
    2. [['this', 'is', 'a', 'list', 'of', 'tokens'], ['some', 'more', 'tokens']]    (e.g. for word similarity)
    By default, each entry in 'for_input' is expected to be a list of preprocessed units (2 above), ready for retrieving a 
    corresponding list of mapped vectors.
    If 'raw==True', however, the strings in the lists in 'for_input' are expected to be unprocessed raw strings, 
    which need to be preprocessed before retrieving a mapped vector.
    In that case, the content of 'for_input' is preprocessed into a list of tokens first.
    This requires that some preprocessing code has been assigned to the respective wec.
    """
    def get_vectors(self,   wec_identifier, 
                            prepro_cache,
                            for_input=[[]], 
                            raw=False, # If True, input will be preprocessed before lookup.
                            default=np.nan, 
                            as_tuple=True, # If True, result will consist of (w,v) tuples, (v) otherwise.
                            in_order=False, # If True, result tuples will be returned in theit org. discourse order, undefined otherwise.
                            ignore_oov=False, # If False, oov words will be replaced by dummy vectors of type <default>, ignored otherwise.
                            no_phrases=False, # If True, phrase detection will be ignored even if phrases were available. 
                            verbose=False):
            
        if len(for_input)==0:
            print("No input to get vectors for! Use get_all_vectors to get all vectors!")
            return

        (we_params_dict_list, _, _, _, _)=expand_parameter_grids(wec_identifier)
        total_result = []

        for we_param_dict in we_params_dict_list:
            we_id = dict_to_sorted_string(we_param_dict, pretty=True)
            if verbose: print(we_id)
            result_for_we = []
            try:
                embdb=self.WM._get_embdb(we_param_dict)
            except NoSuchWombatEmbeddingsException as ex:
                print(ex)
                continue
            if not raw:
                for r in for_input:
                    result_for_we.append(('',r,embdb.get_vectors(prepro_cache, 
                                                    for_input=[r], 
                                                    raw=False, 
                                                    default=default, 
                                                    as_tuple=as_tuple, 
                                                    in_order=in_order, 
                                                    ignore_oov=ignore_oov, 
                                                    no_phrases=no_phrases)))
            else:
                for i in for_input:
                    for line in i:
                        processed=[]
                        if verbose: print(line)
                        # collect raw line plus process output               
                        processed.append((line, embdb.preprocess(line, prepro_cache, no_phrases=no_phrases)))
                        if verbose: print(processed[-1])
                        for (r,p) in processed:
                            result_for_we.append((r,p,embdb.get_vectors(prepro_cache, 
                                                                for_input=[p], 
                                                                raw=False, 
                                                                default=default, 
                                                                as_tuple=as_tuple, 
                                                                in_order=in_order, 
                                                                ignore_oov=ignore_oov, 
                                                                no_phrases=no_phrases)))
                            if verbose: print(result_for_we[-1])
            total_result.append((we_id, result_for_we))
        return total_result
    # end get_vectors


    """
    Get (w,v) tuples for words matching a particular pattern. Empty pattern (default) returns all tuples.
    This uses the 'glob' format described here: http://www.sqlitetutorial.net/sqlite-glob/ 
    """
    def get_matching_vectors(self,
                            wec_identifier, 
                            pattern="",
                            exclude_pattern="",  
                            as_tuple=True, 
                            verbose=False):
        (we_params_dict_list, _, _, _, _)=expand_parameter_grids(wec_identifier)
        total_result = []

        for we_param_dict in we_params_dict_list:
            we_id = dict_to_sorted_string(we_param_dict, pretty=True)
            if verbose: print(we_id)
            embdb=self.WM._get_embdb(we_param_dict)
            vecs=embdb.get_matching_vectors(pattern=pattern, exclude_pattern=exclude_pattern, as_tuple=as_tuple)
            we_id = "P:"+pattern+"_XP:"+exclude_pattern+"@"+we_id
            result_for_we=(we_id,[('',[],vecs)])
            total_result.append(result_for_we)
        return total_result

    def contents(self):
        if self.WM!=None:
            for r in self.WM.DB.cursor().execute("select we_id, dims, unit, fold,descriptor, prepro_name from we_meta order by (we_id)"):
                print(C.BLUE+C.BOLD+str(r)+C.END)    

# end connector class

"""
Wombat Master DB
This class establishes the connection to the wombat master db, which stores meta data for the individual embedding dbs.
The constructor is called by the connector class.
"""
class masterdb(object):
    def __init__(self, path="", create_if_missing=False):
        self.PATH = path # All Embedding DBs will reside in this folder
        if not os.path.exists(self.PATH+MASTER_DB_NAME):
            if create_if_missing:
                print("Creating new DB %s ... "%(self.PATH+MASTER_DB_NAME))
                new_db = sqlite3.connect(self.PATH+MASTER_DB_NAME)
                cursor = new_db.cursor()
                cursor.execute("CREATE TABLE we_meta(we_id INTEGER PRIMARY KEY,dims INTEGER,unit VARCHAR,fold INTEGER,descriptor VARCHAR,\
                prepro_name VARCHAR,prepro_code VARCHAR,CONSTRAINT desc_unique UNIQUE (descriptor))") 
                new_db.commit()
                new_db.close()
            else:
                raise NoSuchWombatMasterDBException("WOMBAT Exception: DB "+self.PATH+MASTER_DB_NAME+" does not exist! Specify 'create_if_missing=True' to create one!")
        self.DB = sqlite3.connect(self.PATH+MASTER_DB_NAME, detect_types=sqlite3.PARSE_DECLTYPES, check_same_thread=False)   
        self.EMBDB_CACHE={}
        return

    def get_prepro_code(self, wec_identifier):
        (prepro_code,) = self.DB.cursor().execute('select prepro_code from we_meta where descriptor=?',(wec_identifier,)).fetchone()        
        return prepro_code

    """
    Try to get name for, open, and return embedding DB identified by we_params_dict from master DB. If it does not exist, 
    a NoSuchWordEmbeddingsException is raised, unless create_new == True, in which case a new DB is created.
    If create_new == True and an embedding DB with we_params_dict already exists in master DB, a DatabaseExistsException
    is raised. 
    If any of the required properties are missing upon DB creation, a MissingRequiredPropertyException is raised.
    Required properties are currently: dims, unit, fold, algo, dataset
    Returns handle to open DB.

    This method is for internal use!

    """
    def _get_embdb(self, we_params_dict, create=False, verbose=False, silent=False):
        embdb=None
        unique_we_descriptor = dict_to_sorted_string(we_params_dict)
        try:                return self.EMBDB_CACHE[unique_we_descriptor]
        except KeyError:    pass

        if create==False:
            # This will raise and pass on a NoSuchWordEmbeddingsException if no DB with descriptor string exists,
            # and if creation is not desired
            # This will be executed upon retrieval
            meta = self._get_embeddingdb_meta(unique_we_descriptor)
            if len(meta) > 0:
                embdb=embeddingdb(meta, verbose=verbose, silent=silent, create=False)
                embdb.set_master(self)
                self.EMBDB_CACHE[unique_we_descriptor] = embdb
                return embdb 
            else:
                raise NoSuchWombatEmbeddingsException("WOMBAT Exception: "+\
                    "No Embedding DB for descriptor "+C.BOLD+C.YELLOW+unique_we_descriptor+C.END)
        else:
            # This will mostly be executed upon import
            dims    =       0
            unit    =       ""      # token, stem, lemma, bpe-XXXX
            fold  =       -1    
            for n in we_params_dict.keys():
                n=n.lower()
                if   n == "dims":   dims=int(we_params_dict[n])
                elif n == "unit":   unit=we_params_dict[n]
                elif n == "fold": fold=int(we_params_dict[n])

            # Check some conditions
            if dims == 0:
                raise(MissingRequiredPropertyException("WOMBAT Exception: "+\
                "No 'dims': value found in provided descriptor "+C.BOLD+C.YELLOW+unique_we_descriptor+C.END)+\
                    "\nPlease specify the dimensionality of the embeddings to be imported!")
            elif unit == "":
                raise(MissingRequiredPropertyException("WOMBAT Exception: "+\
                "No 'unit': value found in provided descriptor "+C.BOLD+C.YELLOW+unique_we_descriptor+C.END)+\
                    "\nPlease specify token, stem, or lemma!")
            elif fold not in [0,1]:
                raise(MissingRequiredPropertyException("WOMBAT Exception: "+\
                "No, or illegal, 'fold': value found in provided descriptor "+C.BOLD+C.YELLOW+unique_we_descriptor+C.END)+\
                    "\nPlease specify 0 or 1!")            

            print("Inserting to WE_META table ...")
            self.DB.cursor().execute('INSERT INTO we_meta(descriptor, dims, unit, fold, prepro_name, prepro_code) values (?,?,?,?,?,?)', (unique_we_descriptor, dims, unit, fold, "", ""))
            self.DB.commit()
            # Get unique DB-internal ID. The corresponding file does not yet exist!
            (we_id,) = self.DB.cursor().execute('Select we_id from we_meta where descriptor=?',(unique_we_descriptor,)).fetchone()        
            print("New embedding DB will get ID %s" %we_id)
            # Create and open DB
            embdb = embeddingdb(self._get_embeddingdb_meta(unique_we_descriptor),verbose=verbose, silent=silent, create=True)
            embdb.set_master(self)
            self.EMBDB_CACHE[unique_we_descriptor] = embdb
        return embdb

    """
    Get meta data for embedding DB identified by we_params_dict from master DB. 
    If it does not exist, a NoSuchWordEmbeddingsException is raised. Otherwise returns a dict consisting of the path, 
    dims, unit, fold, and we_id (the unique number used as prefix to the DB file), and pre-pro information.

    This method is for internal use!

    """
    def _get_embeddingdb_meta(self, unique_we_descriptor):
        resultpath = ""
        dim = 0
        meta = {}
        res_tuple = self.DB.cursor().execute('Select we_id, dims, unit, fold, prepro_name, prepro_code from we_meta where descriptor=?',(unique_we_descriptor,)).fetchone()
        if res_tuple != None:
            (we_id, dims, unit, fold, prepro_name, prepro_code) = res_tuple
            result = self.PATH+"wombat_embs_"+str(we_id)+".sqlite"
            meta={'dims':int(dims), 'unit':unit, 'fold':fold,'unique_we_descriptor':unique_we_descriptor, 'db_path':result, 'prepro_name':prepro_name, 'wombat_path':self.PATH}        
        return meta


"""
Wombat Embedding DB
"""
class embeddingdb(object):
    """ Python wrapper for one sqlite DB containing one set of word embeddings. """
    def __init__(self, meta, verbose=False, silent=False, create=False):
        """ Create an instance with the meta data supplied in dictionary 'meta':
            unique_we_descriptor
            dims
            unit
            fold
            db_path
            prepro_name
            wombat_path
        """
        self.UNIQUE_DESCRIPTOR  = meta['unique_we_descriptor']
        self.DIMS               = int(meta['dims'])
        self.UNIT               = meta['unit']
        self.FOLD               = True if int(meta['fold'])==1 else False
        db_path                 = meta['db_path']
        self.PREPRO_NAME        = meta['prepro_name']
        self.PREPROCESSOR       = None        
        self.MASTER             = None
        wombat_path             = meta['wombat_path']

        if not os.path.exists(db_path):
            if create:
                if not silent: print("Creating new Embedding DB ...")
                new_db = sqlite3.connect(db_path,detect_types=sqlite3.PARSE_DECLTYPES)
                new_db.cursor().execute("CREATE TABLE vectors ( word VARCHAR PRIMARY KEY, vector BLOB )")
                new_db.cursor().execute("CREATE UNIQUE index word_index on vectors (word asc) ")
                new_db.commit()
                new_db.close()
            else:
                raise NoSuchWombatEmbeddingsException("WOMBAT Exception: No Word Embedding DB at\n"+C.BOLD+C.YELLOW+db_path+C.END)
        elif create:
            # Create was set although the db exists already
            raise(WombatEmbeddingDatabaseExistsException("WOMBAT Exception: "+\
                "Word Embedding DB already exists\n"+C.BOLD+C.YELLOW+db_path+C.END))

        if not silent: print(C.BOLD+C.PURPLE+"Connecting to WOMBAT Embedding DB "+C.PURPLE+C.BOLD+db_path+" ... "+C.END)
        self.DB = sqlite3.connect(db_path, detect_types=sqlite3.PARSE_DECLTYPES)
        return

    def set_master(self,master):
        self.MASTER=master

    
    def get_all_vectors(self, as_tuple=True):
        result=[]
        cursor = self.DB.cursor()
        if as_tuple: # Retrieve tuples of (word, vector)
            #query = 'select word, vector from vectors where word in ("the", "a", "and")'
            query = 'select word, vector from vectors'
            result = cursor.execute(query).fetchall()
        else: # Retrieve vectors only
            query = 'select vector from vectors'
            result=cursor.execute(query)
        return result

    def get_vectors(self, prepro_cache, for_input=[[]], raw=True, as_tuple=True, in_order=False, ignore_oov=False, default=np.nan, verbose=False, no_phrases=False):
        """ Get word embedding vectors for the input in for_input. The result is a list of tuples.
            If raw==False, for_input must contain a list of strings that are compatible with the words in the vectors table.
            If raw==True (default), for_input can contain raw, untokenized strings. In that case, the input is preprocessed using the prepro_module associated with this we set.
            as_tuple: Whether the result tuples should contain only *vectors* (False), or *word* and *vector* (True, default)
            in_order: Whether the result tuples should come in the order of the input (True), or in arbitrary order (False, default). In the latter case, at most one result is returned for each different input element.
            ignore_oov: Whether oov input should be ignored (True), or replaced by an array of default values of the correct length (False, default).
            ignore_oov==False works with in_order==True only: If in_order==False, oov input is *always ignored*. 
            (This is related to the optimized way of selection used with in_order==False)

        This method should / need not be called directly, use the equivalent method on the connector class!

        """
        result = []
        if raw:
            for a in for_input:
                for s in a:
                    # Recurse ...
                    # Always use *append* here: flattening was done on the input side already
                    # Make sure to wrap return value of preprocess into a list again
                    c=self.preprocess(s,prepro_cache, verbose=verbose)
                    r =self.get_vectors(prepro_cache, for_input=[c], raw=False,default=default,as_tuple=as_tuple,verbose=verbose,in_order=in_order, ignore_oov=ignore_oov, no_phrases=no_phrases)
                    result.append(r)
                return result
        cursor = self.DB.cursor()
        if in_order==False:
            # Order does not matter
            for a in for_input:
                for i in range(0, len(a), CHUNKSIZE):
                    chunk = a[i:i + CHUNKSIZE]
                    if as_tuple: # Retrieve tuples of (word, vector)
                        query = 'select word, vector from vectors where word in (%s)' % ','.join('?' for i in chunk)
                        for res in cursor.execute(query, chunk):
                            if res != None: 
                                result.append(res)
                            elif not ignore_oov: result.append((t, np.full((self.DIMS,),default)))
                    else: # Retrieve vectors only
                        query = 'select vector from vectors where word in (%s)' % ','.join('?' for i in chunk)
                        for res in cursor.execute(query, chunk):
                            if res != None: 
                                result.append(res)
                            elif not ignore_oov: result.append(np.full((self.DIMS,),default))
        else:
            # Provide vectors in input order. This is somewhat slower than unordered retrieval!!
            for l in for_input:
                for t in l:
                    if as_tuple:
                        res = cursor.execute("select word, vector from vectors where word = ?",(t,)).fetchone()
                        if res != None: 
                            result.append(res)
                        elif not ignore_oov: result.append((t, np.full((self.DIMS,),default)))
                    else:
                        res = cursor.execute("select vector from vectors where word = ?",(t,)).fetchone()
                        if res != None: 
                            result.append(res)
                        elif not ignore_oov: result.append(np.full((self.DIMS,),default))
        if len(result)==0:
            if as_tuple: result.append(('*PAD*', np.full((self.DIMS,),default)))
            else: result.append(np.full((self.DIMS,),default))
        return result


    def get_matching_vectors(self, as_tuple=True, verbose=False, pattern="", exclude_pattern=""):
        result = []
        if as_tuple: # Retrieve tuples of (word, vector)        
            if pattern == "" and exclude_pattern == "":
                # Just get everything
                for res in self.DB.cursor().execute('select word, vector from vectors '):
                    result.append(res)
            elif pattern == "":
                # Everything except exclude_pattern
                for res in self.DB.cursor().execute("select word, vector from vectors where not word glob ?", (exclude_pattern, )):
                    result.append(res)
            elif exclude_pattern == "":
                # Everything matching pattern
                for res in self.DB.cursor().execute("select word, vector from vectors where word glob ?", (pattern, )):
                    result.append(res)                
            else:
                # Everything matching pattern, except except_pattern
                for res in self.DB.cursor().execute("select word, vector from vectors where word glob ?  and not word glob ?", (pattern, exclude_pattern )):
                    result.append(res)                
                                                                    
        else: # Retrieve vectors only
            if pattern == "" and exclude_pattern == "":
                # Just get everything
                for res in self.DB.cursor().execute('select vector from vectors '):
                    result.append(res)
            elif pattern == "":
                # Everything except exclude_pattern
                for res in self.DB.cursor().execute("select vector from vectors where not word glob ?", (exclude_pattern, )):
                    result.append(res)
            elif exclude_pattern == "":
                # Everything matching pattern
                for res in self.DB.cursor().execute("select vector from vectors where word glob ?", (pattern, )):
                    result.append(res)                
            else:
                # Everything matching pattern, except except_pattern
                for res in self.DB.cursor().execute("select vector from vectors where word glob ?  and not word glob ?", (pattern, exclude_pattern )):
                    result.append(res)                
        return result


    def preprocess(self, string, prepro_cache, verbose=False, no_phrases=False):
        if self.PREPROCESSOR!=None:
            try:
                return prepro_cache[string+self.PREPRO_NAME + str(self.FOLD)+str(no_phrases)+self.UNIT]
            except KeyError:
                p=self.PREPROCESSOR.process(string, fold=self.FOLD, conflate=False, unit=self.UNIT, verbose=verbose, no_phrases=no_phrases)
                prepro_cache[string+self.PREPRO_NAME+str(self.FOLD)+str(no_phrases)+self.UNIT]=p
            return p
        elif self.PREPRO_NAME != "":
            # Prepro code is available, but has not been instantiated yet
            self.PREPROCESSOR=pickle.loads(base64.decodestring(self.MASTER.get_prepro_code(self.UNIQUE_DESCRIPTOR)))
            return self.preprocess(string, prepro_cache, verbose=verbose, no_phrases=no_phrases)
        else:
            print("No preprocessing code available")
            return string.split(" ")


