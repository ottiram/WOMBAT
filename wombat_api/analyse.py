import sys, os, numpy
from textwrap import wrap
import matplotlib.pyplot as plt
#from matplotlib.backends.backend_pdf import PdfPages
from operator import itemgetter
import scipy.spatial.distance as dist
from collections import Counter
from sklearn import manifold
from tqdm import tqdm

from wombat_api.lib import *
TITLE_WRAP=40

def pairwise_distances(input_1, input_2, metric=dist.cosine, up_to_index=100000, reverse=False, verbose=False):
    total_result=[]
    assert len(input_1) == len(input_2)
    for we_count in range(len(input_1)):
        (wec_name, s1_content)  = input_1[we_count]
        results_for_wec=[]
        (_, s2_content)         = input_2[we_count]
        assert len(s1_content) == len(s2_content)
        limit=min(up_to_index, len(s1_content)-1)
        #print("Looking only up to list pos %s"%limit)
        for sequence_count in range(len(s1_content)):
            if sequence_count>limit: break
            s1_sequence=s1_content[sequence_count]
            s1_string=s1_sequence[0]
            s1_tokens=s1_sequence[1]
            s1_vectors=[x[1] for x in s1_sequence[2]]
            s2_sequence=s2_content[sequence_count]
            s2_string=s2_sequence[0]
            s2_tokens=s2_sequence[1]
            s2_vectors=[x[1] for x in s2_sequence[2]]
            s1_avg = np.average(s1_vectors, axis=0)
            s2_avg = np.average(s2_vectors, axis=0)
            results_for_wec.append((float(metric(s1_avg, s2_avg)), s1_string, s1_tokens, s2_string, s2_tokens))            
        total_result.append(sorted(results_for_wec, key=itemgetter(0), reverse=reverse))
        total_result[-1].insert(0,metric.__name__)
        total_result[-1].insert(0,wec_name)
    return total_result

def cartesian_distances(input_1, metric=dist.cosine, up_to_index=100000, reverse=False, verbose=False):
    total_result=[]
    for we_count in range(len(input_1)):
        (wec_name, content)  = input_1[we_count]
        results_for_wec=[]
        limit=min(up_to_index,len(content)-1)
        #print("Looking only up to list pos %s"%limit)
        for o, s1_sequence in enumerate(content):
            if o>limit: break
            s1_string=s1_sequence[0]
            s1_tokens=s1_sequence[1]
            s1_vectors=[x[1] for x in s1_sequence[2]]
            s1_avg = np.average(s1_vectors, axis=0)
            for i in range (o+1,len(content)):
                if i>limit: break
                s2_sequence=content[i]
                s2_string=s2_sequence[0]
                s2_tokens=s2_sequence[1]
                s2_vectors=[x[1] for x in s2_sequence[2]]
                s2_avg = np.average(s2_vectors, axis=0)
                results_for_wec.append((float(metric(s1_avg, s2_avg)), s1_string, s1_tokens, s2_string, s2_tokens))            
        total_result.append(sorted(results_for_wec, key=itemgetter(0), reverse=reverse))
        total_result[-1].insert(0,metric.__name__)
        total_result[-1].insert(0,wec_name)
    return total_result

def plot_heatmap(
    tuples1, tuples2,
    metric=dist.cosine, 
    string1="",
    string2="",
    verbose=False, 
    ignore_matching=False, 
    plot_name="", 
    cmap="RdYlGn", 
    default=0.0, 
    suptitle_props={'fontsize':12, 'fontweight':'bold'}, 
    plottitle_props={'fontsize':12, 'fontweight':'normal'}, 
    ticklabel_props={'fontsize':12, 'fontweight':'bold'},#
    title=""):
    if plot_name=="":
        plot_name="heatmap-"+str(os.getpid())+".png"
        
    w=max(len(tuples1),4)
    b=max(len(tuples2),4)

    # Create contents for current page
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(w,b), squeeze=False)
    (matrix, xwords, ywords) = distance_matrix(tuples1, tuples2, metric=metric, ignore_matching=ignore_matching, invert=True)
    if matrix!=[]:
#        t=("Measure: "+metric.__name__.upper()+"\n\n")+"\n".join(wrap(title, TITLE_WRAP))
#        fig.suptitle(t, **suptitle_props)
        heatplot = axes[0,0].imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(heatplot, ax=axes[0,0])
        axes[0,0].set_xticks(range(len(xwords)))
        axes[0,0].set_xticklabels(xwords, rotation=90, **ticklabel_props)
        axes[0,0].set_xlabel("\n".join(wrap(string1, 500)), fontsize=14, fontweight="bold")
        axes[0,0].set_ylabel("\n".join(wrap(string2, 500)), fontsize=14, fontweight="bold")
        axes[0,0].set_yticks(range(len(ywords)))
        axes[0,0].set_yticklabels(ywords, **ticklabel_props)

        # Loop over data dimensions and create text annotations.
        for i in range(len(ywords)):
            for j in range(len(xwords)):
                try:
                    c="white"
                    if matrix[i, j] >= 0.3 and matrix[i, j] <= 0.7: c="black"
                    axes[0,0].text(j, i, '{0:.3f}'.format(matrix[i, j]), ha="center", va="center", color=c, fontweight="bold")
                except IndexError:
                    pass
        plt.tight_layout()
        plt.savefig(plot_name)
    plt.close()

def distance_matrix(tuples1, tuples2, metric=dist.cosine, ignore_matching=True, invert=False, ignore=['*sw*']):
        matrix,xwords,ywords=[],[],[]
        ignorable = numpy.intersect1d([v[0] for v in tuples1], [v[0] for v in tuples2]) if ignore_matching else []
        ignorable.extend(ignore)
        # Remove all ignorable items from both lists first
        # This might be inefficient ... 
        to_del1, to_del2=[],[]
        for w in ignorable:
            for d in range(len(tuples1)):
                if w == tuples1[d][0]: to_del1.append(d)
            for d in range(len(tuples2)):
                if w == tuples2[d][0]: to_del2.append(d)
        for i in reversed(to_del1):
            del tuples1[i]
        for i in reversed(to_del2):
            del tuples2[i]

        rownum=len(tuples2)-1
        while rownum>=0:
            row=[]
            (word2,vector2) = tuples2[rownum]
            if numpy.isnan(vector2).any() or word2 in ignorable:
                rownum-=1
                continue
            ywords.append(word2)
            for colnum in range(len(tuples1)):
                (word1,vector1) = tuples1[colnum]
                if numpy.isnan(vector1).any() or word1 in ignorable:
                    continue
                if len(xwords) < len(tuples1):#-len(ignorable)):
                    xwords.append(word1)
                if invert:  
                    row.append(1-metric(vector1,vector2))
                else:
                    row.append(metric(vector1,vector2))
            matrix.append(row)
            rownum-=1
        return ((np.array(matrix), xwords, ywords))
"""
def distance_matrix(tuples1, tuples2, metric=dist.cosine, ignore_matching=True, invert=False, ignore=['*sw*']):
        matrix,xwords,ywords=[],[],[]
        matching = numpy.intersect1d([v[0] for v in tuples1], [v[0] for v in tuples2]) if ignore_matching else []
        matching.extend(ignore)
        rownum=len(tuples2)-1
        while rownum>=0:
            row=[]
            (word2,vector2) = tuples2[rownum]
            if numpy.isnan(vector2).any() or word2 in matching:
                rownum-=1
                continue
            ywords.append(word2)
            for colnum in range(len(tuples1)):
                (word1,vector1) = tuples1[colnum]
                if numpy.isnan(vector1).any() or word1 in matching:
                    continue
                if len(xwords) < (len(tuples1)-len(matching)): xwords.append(word1)
                if invert:  
                    row.append(1-metric(vector1,vector2))
                else:
                    row.append(metric(vector1,vector2))
            matrix.append(row)
            rownum-=1
        print(xwords)
        return ((np.array(matrix), xwords, ywords))
"""

"""
def plot_heatmap(wb, 
    we_param_grid_string, 
    for_input_pair=([[]], [[]]), 
    measure=dist.cosine, 
    verbose=False, 
    ignore_matching=False, 
    ignore_oov=False, 
    raw=True, 
    figsize=(5,5), 
    pdf_name="", 
    cmap="RdYlGn", 
    default=0.0, 
    suptitle_props={'fontsize':12, 'fontweight':'bold'}, 
    plottitle_props={'fontsize':12, 'fontweight':'normal'}, 
    ticklabel_props={'fontsize':12, 'fontweight':'bold'}):

    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(we_param_grid_string)
    pdf=None
    # Open prior to page processing,
    if pdf_name!="":                # if output name was supplied
        pdf=PdfPages(pdf_name)
    elif plot_pages > 1:            # or if we have three grids, which we cannot plot to the screen
        pdf=PdfPages("heatmap-"+str(os.getpid())+".pdf")
    # Iterate over pages (is at least one)
    for pages in range(plot_pages):
        # Create contents for current page
        fig, axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(figsize[0]*plot_cols, figsize[1]*plot_rows), squeeze=False)
        t=("Measure: "+measure.__name__.upper()+"\n\n")+"\n".join(wrap(we_param_grid_string.replace(";","; "), TITLE_WRAP))
        fig.suptitle(t, **suptitle_props)
        for p,we_params_dict in enumerate(we_params_dict_list):
            (row,col,page)=plot_coords[p]
            if pages != page: continue
            emb_db = wb.WM.get_emb_db(we_params_dict)
            
            (matrix, xwords, ywords) = emb_db.get_matrix(for_input_pair=for_input_pair, verbose=verbose, ignore_matching=ignore_matching, raw=raw, ignore_oov=ignore_oov, default=default)
            
            heatplot = axes[row,col].imshow(matrix, cmap=cmap)
            plt.colorbar(heatplot, ax=axes[row,col])
            name=dict_to_sorted_string(we_params_dict)
            axes[row,col].set_title("\n".join(wrap(name, TITLE_WRAP)), **plottitle_props)
            axes[row,col].set_xticks(range(len(xwords)))
            axes[row,col].set_xticklabels(xwords, rotation=90, **ticklabel_props)
            axes[row,col].set_yticks(range(len(ywords)))
            axes[row,col].set_yticklabels(ywords, **ticklabel_props)
            plt.tight_layout()
            emb_db.DB.close()
        if pdf != None: pdf.savefig()
    if pdf != None: pdf.close()
"""


############################
           
def plot_nearest_neighbors(wb, 
    we_param_grid_string, 
    target, 
    count=10, 
    measure=dist.cosine, 
    to_rank=[[]], 
    raw_to_rank=False,
    verbose=False, 
    pdf_name="", 
    figsize=(5,5), 
    suptitle_props= {'fontsize':12, 'fontweight':'bold'},
    plottitle_props={'fontsize':12, 'fontweight':'bold'},
    textitem_props= {'fontsize':12, 'fontweight':'bold', 'color':'black'},
    ticklabel_props={'labelsize':12},
    bar_props={'color':'lightgreen'},
    unique_item_color="red",
    common_item_color="white"):    
    
    """ 
    Return from the word embeddings specified by 'we_param_grid_string' the 'count' (default 10) items 
    most similar to 'target', from most to least similar (=least to most distant). 
    Use two-place function 'measure' (default scipy.spatial.distance.cosine) for computing the similarity.
    If 'to_rank' is specified, only its similarity to 'target' is computed and returned, sorted from most to least similar.
    Returns a list of <result, we_desc> tuples, where result is itself a list of <word, sim> tuples.
    """
    
    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(we_param_grid_string)
    results = []
    word_counter = Counter()

    # Iterate over all supplied we sets and create results for each
    for we_params_dict in we_params_dict_list:
        # TODO: Move most_similar_plain to wombat_embedding_db.py
        most_sim = nearest_neighbors(wb, dict_to_sorted_string(we_params_dict,pretty=True), target, count=count, measure=measure, to_rank=to_rank, raw_to_rank=raw_to_rank, verbose=verbose)[0][0]        
        results.append(most_sim)
        for tup in most_sim: word_counter[tup[0]]+=1    
    try:
        default_color=textitem_props['color']
    except KeyError:
        default_color='black'
    pdf=None
    # Open prior to page processing,
    if pdf_name!="":                # if output name was supplied
        pdf=PdfPages(pdf_name)
    elif plot_pages > 1:            # or if we have three grids, which we cannot plot to the screen
        pdf=PdfPages("nn-"+str(os.getpid())+".pdf")
                
    # Iterate over pages (there is at least one)
    for pages in range(plot_pages):
        # Create contents for current page
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(figsize[0]*plot_cols, figsize[1]*plot_rows), squeeze=False)

        t=("Target word: '"+target+"', Measure: "+measure.__name__.upper()+"\n")+"\n".join(wrap(we_param_grid_string.replace(";","; "), TITLE_WRAP))
        fig.suptitle(t, **suptitle_props)

        #fig.suptitle("Target word: '"+target+"', measure: "+(measure.__name__.upper())+"\n"+we_param_grid_string, **suptitle_props)
        for p,most_sim in enumerate(results):
            (row,col,page)=plot_coords[p]
            if pages != page: continue            
            name=dict_to_sorted_string(we_params_dict_list[p])
            axes[row,col].set_title("\n".join(wrap(name, TITLE_WRAP)), **plottitle_props)            
            axes[row,col].set_xticks([])
            axes[row,col].tick_params(**ticklabel_props)
            dists, dist_nums=[],[]
            # Dummy, without this, no .patches will be available. 
            axes[row,col].barh(range(count),np.zeros(count),**bar_props)            
            for rect, sim_tup in zip(axes[row,col].patches, most_sim):
                word = sim_tup[0]
                 # default: in between
                if word_counter[word] == 1:                 textitem_props['color']=unique_item_color
                elif word_counter[word] == len(results) :   textitem_props['color']=common_item_color
                else:                                       textitem_props['color']=default_color
                axes[row,col].text(rect.get_x(), rect.get_y()+0.4, "  "+word, ha='left', va='center', **textitem_props)
                dists.append('{:01.4f}'.format(sim_tup[1]))
                dist_nums.append(sim_tup[1])
            axes[row,col].set_ylim(count,-1)
            axes[row,col].set_yticks(range(count))
            axes[row,col].set_yticklabels(dists)                
            axes[row,col].barh(range(count),dist_nums,**bar_props)                
        if pdf != None: pdf.savefig()
        #if plot_pages == 1: plt.show()
    #end of page
    textitem_props['color']=default_color
    if pdf != None: pdf.close()
            

def nearest_neighbors(wb, we_param_grid_string, target, count=10, measure=dist.cosine, to_rank=[[]], raw_to_rank=False, verbose=False):
    (we_params_dict_list,_ ,_ ,_ ,_) = expand_parameter_grids(we_param_grid_string)
    all_results=[]
    for we_params_dict in we_params_dict_list:
        emb_db = wb.WM.get_emb_db(we_params_dict)
        # Get tuple to compare to. 
        target_tuple = emb_db.get_vectors(for_input=[[target]], raw=False, as_tuple=True, default=np.nan, verbose=verbose)
        if np.isnan(target_tuple[0][1][0]):
            print("Target '%s' not found in '%s'"%(target,dict_to_sorted_string(we_params_dict, pretty=True)))
            continue
        current_dist=float(0.0)
        result = []        
        if len(to_rank[0])>0:
            processed_input_set=set()
            if raw_to_rank:
                for a in to_rank:
                    for line in a:
                        r=emb_db.preprocess(line,verbose=verbose)
                        processed_input_set.update(r)
            else:
                processed_input_set=set(to_rank[0])
            retrieved=emb_db.get_vectors(for_input=[list(processed_input_set)], raw=False, as_tuple=True, verbose=verbose)
        else:
            retrieved=emb_db.DB.cursor().execute('Select word, vector from VECTORS')
        for row in retrieved:
            # Each row is a flat (w,v) tuple
            if row[0] == target: continue
            current_dist = float(measure(target_tuple[0][1], row[1]))
            if len(result) < count:
                # Fill result list to required length
                result.append((row[0], current_dist))
            else:
                # The list is full already, assume ordering from least to most dist
                if current_dist < result[-1][1]:
                    # The current dist is less than the previous
                    result.append((row[0], current_dist))
                    result=sorted(result, key=itemgetter(1))
                    result=result[:count]
        # Sort once more in case we never found 'count' items
        result=sorted(result, key=itemgetter(1))
        all_results.append((result,dict_to_sorted_string(we_params_dict,pretty=True)))
    return all_results



def tsne(wb, we_param_grid_string, for_matches="", for_input=[[]], string1="", items=[], highlight=[], pdf_name="", iters=250, figsize=(50,50), share_axes=('all','all'),fontsize=14):
    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(we_param_grid_string)
    pdf=None
    # Open prior to page processing,
    if pdf_name!="":                # if output name was supplied
        pdf=PdfPages(pdf_name)
    elif plot_pages > 1:            # or if we have three grids, which we cannot plot to the screen
        pdf=PdfPages("tsne-"+str(we_param_dist_list)+".pdf")
    # Iterate over pages (is at least one)
    for pages in range(plot_pages):
        # Create contents for current page
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(figsize[0]*plot_cols, figsize[1]*plot_rows), squeeze=False, sharex=share_axes[0], sharey=share_axes[1])
        for p,we_params_dict in enumerate(we_params_dict_list):
            (row,col,page)=plot_coords[p]
            if pages != page: continue
            axes[row,col].tick_params(labelsize=14)
            name=dict_to_sorted_string(we_params_dict)
            axes[row,col].set_title("\n".join(wrap(name, 60)), fontweight='bold', fontsize=14)
            try: emb_db = wb.WM.get_embdb(we_params_dict)
            except NoSuchWordEmbeddingsException:
                #name=dict_to_sorted_string(we_params_dict)
                #axes[row,col].set_title("\n".join(wrap(name, 60)), fontweight='bold', fontsize=14)
                result=[]                
                print("N/A, ignored!")
                miny, maxy = axes[row,col].get_ylim()
                minx, maxx = axes[row,col].get_xlim()
                axes[row,col].text((minx+maxx)/2, (miny+maxy)/2, "N/A", fontsize=40, ha='center', va='center')
                continue
#            if string1 != "":
            if for_matches != "":
                result=emb_db.get_all_vectors(for_matches=for_matches, as_tuple=True)
            else:
                result=emb_db.get_vectors({}, for_input=for_input, as_tuple=True, default=float(0.0), raw=True)
            #else:
            #    result=emb_db.get_vectors({}, for_input=items, as_tuple=True, default=float(0.0), raw=False)
            #result=emb_db.get_all_vectors(as_tuple=True)
            if len(result) > 0:
                #name=dict_to_sorted_string(we_params_dict)
                #axes[row,col].set_title("\n".join(wrap(name, 60)), fontweight='bold', fontsize=14)
                # result is list of lists if get_vectors was called with for_raw_strings
                flattened_result=[]
                unifier=set()
                if type(result[0])== list: # Result is a list of lists, which happens if input is unprocessed strings
                    for r in result: 
                        for v in r:
                            # Add each (w,v) tuple in result only once
                            if v[0] not in unifier:
                                flattened_result.append(v)
                                unifier.add(v[0])
                else:
                    flattened_result = result
                units, vectors = zip(*flattened_result)
                print("%s units to plot"%len(units))
                x = np.asarray(vectors)
                sys.stdout.write("Doing tsne magic ...")
                sys.stdout.flush()
                tsne = manifold.TSNE(n_components=2, init='pca', random_state=4711, n_iter=iters)
                y = tsne.fit_transform(x)
                sys.stdout.write(" done\n")
                sys.stdout.flush()
                print("Plotting")
                b = tqdm(total=len(units), ncols=PROGBARWIDTH)
                axes[row,col].scatter(y[:, 0], y[:, 1], marker='.', c='w')
                for i, txt in enumerate(units):
                    if txt in highlight:
                        axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize+4, color='orange', fontweight='bold')        
                    else:
                        axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='black', fontweight='normal')
                    b.update(1)
                b.close()
            emb_db.DB.close()
        if pdf != None: pdf.savefig()
        if plot_pages == 1: plt.show()
    #end of page
    if pdf != None: pdf.close()


def sentence_similarity(wb, we_param_grid_string="", pairs=[], figsize=(50,50), share_axes=('all','none'), max_pairs=20, pdf_name="", verbose=False, textsize=12):
    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(we_param_grid_string)
    pdf=None
    # Open prior to page processing,
    if pdf_name!="":                # if output name was supplied
        pdf=PdfPages(pdf_name)
    elif plot_pages > 1:            # or if we have three grids, which we cannot plot to the screen
        pdf=PdfPages("plots/sent-sim_"+we_param_grid_string+".pdf")

    max_dist=0.0
    for pages in range(plot_pages):
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, figsize=(figsize[0]*plot_cols, figsize[1]*plot_rows), squeeze=False, sharex=share_axes[0], sharey=share_axes[1])
        for p,we_params_dict in enumerate(we_params_dict_list):
            (row,col,page)=plot_coords[p]
            if pages != page: continue

            axes[row,col].set_ylim(max_pairs,-1)
            axes[row,col].set_xlim(0,0.03)
            axes[row,col].set_yticks(np.arange(max_pairs))
            # axes[row,col].set_xticks(np.arange(0,0.03,0.005))
            axes[row,col].tick_params(labelsize=10)

            try:
                emb_db = wb.WM.get_emb_db(we_params_dict)
            except NoSuchWordEmbeddingsException:
                print("N/A, ignored!")
                name=dict_to_sorted_string(we_params_dict)
                axes[row,col].set_title("\n".join(wrap(name, 60)), fontweight='bold', fontsize=14)
                miny, maxy = axes[row,col].get_ylim()
                minx, maxx = axes[row,col].get_xlim()
                axes[row,col].text((minx+maxx)/2, (miny+maxy)/2, "N/A", fontsize=40, ha='center', va='center')
                continue
            results=[]

            name=dict_to_sorted_string(we_params_dict)
            axes[row,col].set_title("\n".join(wrap(name, 60)), fontweight='bold', fontsize=14)

            s1_raw_strings, s2_raw_strings = [],[]
            for i,(s1, s2) in enumerate(pairs):
                s1_raw_strings.append(s1)
                s2_raw_strings.append(s2)

            # Using for_raw_string causes the emb_db-related tokenizer / stemmer / stopper to be invoked                
            s1_vectors = emb_db.get_vectors_bulk(for_raw_strings=s1_raw_strings, as_tuple=False, verbose=verbose)
            s2_vectors = emb_db.get_vectors_bulk(for_raw_strings=s2_raw_strings, as_tuple=False, verbose=verbose)
            assert len(s1_vectors) == len(s2_vectors)
            for i in range (len(s1_vectors)):        
                s1_avg = np.average(s1_vectors[i], axis=0)
                s2_avg = np.average(s2_vectors[i], axis=0)
                results.append((i, s1_raw_strings[i], s2_raw_strings[i], float(dist.cosine(s1_avg, s2_avg))))

            # Sort by cos distance
            data=sorted(results, key=itemgetter(3), reverse=False)
            data=data[:max_pairs]
            pair_ids, distances,distance_labels,sentences=[],[],[],[]
            pair_pos = np.arange(max_pairs)
            for d in data:
                pair_ids.append("{:03d}".format(d[0])+" ({0:0.5f})".format(d[3]))
                distances.append(d[3])
                sentences.append(d[1]+" <--> "+d[2])                                
            max_dist = max(max_dist,float(distances[-1]))
            axes[row,col].barh(pair_pos,distances,color='lightgreen')
            axes[row,col].set_yticklabels(pair_ids)
            
            for rect, sentence in zip(axes[row,col].patches, sentences):
                axes[row,col].text(rect.get_x()+0.0001, rect.get_y()+0.4, sentence, ha='left', va='center', fontsize=textsize)
            emb_db.DB.close()
        plt.xlim(0,max_dist*1.1)
        if pdf != None: pdf.savefig()
        if plot_pages == 1: plt.show()
    #end of page
    if pdf != None: pdf.close()



def get_pairwise_tuples(self, prepro_cache, input1=[], input2=[], raw=False, metric_names=['cosine'], verbose=False, ignore_matching=False, default=np.nan, ignore_oov=True, ignore=[], idf_dict=None):
    results_per_metric={}
    metric_comps=[]
    # metric_names is expected to be unique
    for m in metric_names:
        try:
            metric_comps.append(METRICS[m])
            # Create empty list to collect results for metric m
            results_per_metric[m]=[]
        except KeyError:
            print("No distance metric %s, ignored!"%measure)

    result, xwords, ywords=[],[],[]
    tups1=self.get_vectors(prepro_cache, for_input=[input1], as_tuple=True, default=default, raw=raw, verbose=verbose, in_order=False, ignore_oov=ignore_oov)
    tups2=self.get_vectors(prepro_cache, for_input=[input2], as_tuple=True, default=default, raw=raw, verbose=True, in_order=False, ignore_oov=ignore_oov)
#        print(tups1)

    for t1_index in range(len(tups1)):
        (word1,vector1) = tups1[t1_index]
        if np.isnan(vector1).any() or word1 in ignore: continue
        try:    idf1=idf_dict[word1]
        except: idf1=np.nan
        xwords.append(word1)
        for t2_index in range(len(tups2)):
            (word2,vector2) = tups2[t2_index]
            if np.isnan(vector2).any() or word2 in ignore: continue
            try:    idf2=idf_dict[word2]
            except: idf2=np.nan
            ywords.append(word2)
            # Compute all distances for current pair
            for metric_comp in metric_comps:
                results_per_metric[metric_comp.__name__].append((metric_comp(vector1,vector2), word1, idf1, word2, idf2))
    return (results_per_metric, xwords, ywords)





