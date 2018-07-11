import sys, os, numpy, re, psutil
import webbrowser as wb
from operator import itemgetter
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import manifold
import scipy.spatial.distance

from wombat_api.lib import *
TITLE_WRAP=40

def plot_pairwise_distances(vector_result1, vector_result2, pdf_name="", size=(10,10), share_axes=('all','none'), max_pairs=20,  verbose=False, fontsize=14, metric=scipy.spatial.distance.cosine, arrange_by="", silent=False):
    
    cartesian_mode=False

    if vector_result2 == None:
        print("Creating cartesian product of single input list!")
        cartesian_mode=True

    if not cartesian_mode:
        # Each input must contain data from the same number of WECs (optimally just one)
        assert len(vector_result1) == len(vector_result2)

    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(arrange_by)
    
    pdf_name=pdf_name if pdf_name!="" else str(psutil.Process().pid)+"_plot.pdf"
    pdf=PdfPages(pdf_name)

    # Iterate over pages (there is at least one)
    for pages in range(plot_pages):
        # Create container for current page
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, 
                        figsize=(size[0]*plot_cols, size[1]*plot_rows), 
                        squeeze=False, sharex=share_axes[0], sharey=share_axes[1])
        for p,we_params_dict in enumerate(we_params_dict_list):
            # p is also the index into vector_result1 and 2
            (row,col,page)=plot_coords[p]
            if pages != page: continue
            max_dist=0.0
            axes[row,col].set_ylim(max_pairs,-1)
            axes[row,col].set_xlim(0,0.03)
            axes[row,col].set_yticks(np.arange(max_pairs))
            axes[row,col].tick_params(labelsize=10)
            name=metric.__name__.upper()+": "+dict_to_sorted_string(we_params_dict)
            axes[row,col].set_title("\n".join(wrap(name, 100)), fontweight='bold', fontsize=14)

            if not cartesian_mode:
                # f1 and f2 are complete results for WEC p
                f1=vector_result1[p][1]
                f2=vector_result2[p][1]
                # Same no. of items in list to compare
                assert len(f1) == len(f2)
                results=[]
                for t in range(len(f1)):

                    # Todo Make this more efficient
                    vecs1,vecs2=[],[]
                    for v in f1[t][2]:
                        vecs1.append(v[1])
                    for v in f2[t][2]:
                        vecs2.append(v[1])

                    s1_avg = np.average(vecs1, axis=0)
                    s2_avg = np.average(vecs2, axis=0)
                    results.append((t, f1[t][0], f2[t][0], float(metric(np.average(vecs1, axis=0), np.average(vecs2, axis=0)))))
            else:
                # f1 is the complete result for WEC p
                f1=vector_result1[p][1]
                temp,results=[],[]
                for t in range(len(f1)):
                    vecs1=[]
                    # sent1 = f1[t][0]
                    for v in f1[t][2]:
                        vecs1.append(v[1])
                    s1_avg = np.average(vecs1, axis=0)
                    temp.append((np.average(vecs1, axis=0), f1[t][0]))

                c=0
                for outer in range(len(temp)):
                    s1_avg=temp[outer][0]
                    sent1=temp[outer][1]
                    for inner in range(outer+1,len(temp)):
                        s2_avg=temp[inner][0]
                        sent2=temp[inner][1]
                        c+=1
                        results.append((c, sent1, sent2, float(metric(s1_avg, s2_avg))))

            # Sort by distance 
            data=sorted(results, key=itemgetter(3), reverse=False)
            data=data[:max_pairs]
            pair_ids, distances, distance_labels, sentences=[],[],[],[]
            pair_pos = np.arange(max_pairs)
            for d in data:
                pair_ids.append("{:03d}".format(d[0])+" ({0:0.5f})".format(d[3]))
                distances.append(d[3])
                sentences.append(d[1]+" <--> "+d[2])                                
            max_dist = max(max_dist,float(distances[-1]))
            axes[row,col].barh(pair_pos,distances,color='lightgreen')
            axes[row,col].set_yticklabels(pair_ids)
            
            for rect, sentence in zip(axes[row,col].patches, sentences):
                axes[row,col].text(rect.get_x()+0.0001, rect.get_y()+0.4, sentence, ha='left', va='center', fontsize=fontsize)
        # plt.tight_layout()
        plt.xlim(0,max_dist*1.1)
        pdf.savefig()
    pdf.close()
    if not silent: wb.open(pdf_name)



def plot_tsne(vector_result, pdf_name="", iters=250, size=(10,10), share_axes=('none','none'), fontsize=14, arrange_by="", highlight="", silent=True):
    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(arrange_by)
    
    pdf_name=pdf_name if pdf_name!="" else str(psutil.Process().pid)+"_plot.pdf"
    pdf=PdfPages(pdf_name)

    # Iterate over pages (there is at least one)
    for pages in range(plot_pages):
        # Create container for current page
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, 
                        figsize=(size[0]*plot_cols, size[1]*plot_rows), 
                        squeeze=False, sharex=share_axes[0], sharey=share_axes[1])
        for p,we_params_dict in enumerate(we_params_dict_list):
            (row,col,page)=plot_coords[p]
            if pages != page: continue
            vectors,units=[],[]
            for (_, _, tuples) in vector_result[p][1]:
                for (w,v) in tuples:
                    units.append(w)
                    vectors.append(v)
            x = np.asarray(vectors)
            sys.stdout.write("Doing tsne magic ...")
            sys.stdout.flush()
            tsne = manifold.TSNE(n_components=2, init='pca', random_state=4711, n_iter=iters)
            y = tsne.fit_transform(x)
            sys.stdout.write(" done\n")
            sys.stdout.flush()
            axes[row,col].scatter(y[:, 0], y[:, 1], marker='.', c='w')
            axes[row,col].set_title(vector_result[p][0])
            for i, txt in enumerate(units):
                if highlight=="":
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='black', fontweight='normal')
                elif re.search(highlight, txt)!=None:
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='green', fontweight='bold')
                else:
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='black', fontweight='normal')
        pdf.savefig()    
    pdf.close()
    if not silent: wb.open(pdf_name)


def compute_distance_matrix(vector_result1, vector_result2, metric=scipy.spatial.distance.cosine, ignore_matching=True, invert=False, ignore=['*sw*']):
    # Each input must contain data from the same number of WECs (optimally just one)
    assert len(vector_result1) == len(vector_result2)

    tuples1,tuples2,matrix,xwords,ywords,result=[],[],[],[],[],[]
    # Iterate over results for each WEC
    for p in range (len(vector_result1)):
        result_for_wec=[]
        f1=vector_result1[p][1]
        f2=vector_result2[p][1]

        assert len(f1) == len(f2)
        # Iterate over results for all sentences, creating pairs
        for t in range(len(f1)):
            matrix,xwords,ywords=[],[],[]
            string1=f1[t][0]
            string2=f2[t][0]
            tuples1=f1[t][2]
            tuples2=f2[t][2]

            ignorable = numpy.intersect1d([v[0] for v in tuples1], [v[0] for v in tuples2]).tolist() if ignore_matching else []
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
            result_for_wec.append((np.array(matrix), xwords, ywords, string1, string2))
        result.append(result_for_wec)
    return result

def plot_heatmap(matrix, xwords, ywords, xstring="", ystring="", plot_name="", cmap="RdYlGn", default=0.0, title="",
    suptitle_props={'fontsize':12, 'fontweight':'bold'}, 
    plottitle_props={'fontsize':10, 'fontweight':'normal'}, 
    ticklabel_props={'fontsize':10, 'fontweight':'normal'},
    verbose=False):
    if plot_name=="":
        plot_name="heatmap-"+str(os.getpid())+".png"
        
    w=max(len(xwords),4)
    b=max(len(ywords),4)

    # Create contents for current page
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(w,b), squeeze=False)
    if matrix!=[]:
        t="".join(wrap(title, TITLE_WRAP)) if title!="" else ""
        fig.suptitle(t, **suptitle_props)
        heatplot = axes[0,0].imshow(matrix, cmap=cmap, vmin=0, vmax=1)
        plt.colorbar(heatplot, ax=axes[0,0])
        axes[0,0].set_xticks(range(len(xwords)))
        axes[0,0].set_xticklabels(xwords, rotation=90, **ticklabel_props)
        axes[0,0].set_xlabel("\n".join(wrap(xstring, 40*(w/4))), fontsize=10, fontweight="normal")
        axes[0,0].set_ylabel("\n".join(wrap(ystring, 40*(w/4))), fontsize=10, fontweight="normal")
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


#def compute_pairwise_distances(input_1, input_2, metric=dist.cosine, up_to_index=100000, reverse=False, verbose=False):
#    total_result=[]
#    assert len(input_1) == len(input_2)
#    for we_count in range(len(input_1)):
#        (wec_name, s1_content)  = input_1[we_count]
#        results_for_wec=[]
#        (_, s2_content)         = input_2[we_count]
#        assert len(s1_content) == len(s2_content)
#        limit=min(up_to_index, len(s1_content)-1)
#        #print("Looking only up to list pos %s"%limit)
#        for sequence_count in range(len(s1_content)):
#            if sequence_count>limit: break
#            s1_sequence=s1_content[sequence_count]
#            s1_string=s1_sequence[0]
#            s1_tokens=s1_sequence[1]
#            s1_vectors=[x[1] for x in s1_sequence[2]]
#            s2_sequence=s2_content[sequence_count]
#            s2_string=s2_sequence[0]
#            s2_tokens=s2_sequence[1]
#            s2_vectors=[x[1] for x in s2_sequence[2]]
#            s1_avg = np.average(s1_vectors, axis=0)
#            s2_avg = np.average(s2_vectors, axis=0)
#            results_for_wec.append((float(metric(s1_avg, s2_avg)), s1_string, s1_tokens, s2_string, s2_tokens))            
#        total_result.append(sorted(results_for_wec, key=itemgetter(0), reverse=reverse))
#        total_result[-1].insert(0,metric.__name__)
#        total_result[-1].insert(0,wec_name)
#    return total_result




