import sys, os, numpy, re, psutil
import webbrowser as wb
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import manifold
import scipy.spatial.distance as dist

from wombat_api.lib import *
TITLE_WRAP=40

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


def compute_distance_matrix(vector_result1, vector_result2, metric=dist.cosine, ignore_matching=True, invert=False, ignore=['*sw*']):
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
            result_for_wec.append((np.array(matrix), xwords, ywords))
        result.append(result_for_wec)
    return result

def plot_heatmap(matrix, xwords, ywords, string1="", string2="", verbose=False, plot_name="",  cmap="RdYlGn", default=0.0, 
    suptitle_props={'fontsize':12, 'fontweight':'bold'}, 
    plottitle_props={'fontsize':12, 'fontweight':'normal'}, 
    ticklabel_props={'fontsize':12, 'fontweight':'bold'},
    title=""):
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




