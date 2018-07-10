import sys, os, numpy, re, psutil
import webbrowser as wb
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import manifold

#from textwrap import wrap
#from operator import itemgetter
#import scipy.spatial.distance as dist
#from collections import Counter
#from tqdm import tqdm

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







