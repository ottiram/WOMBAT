import sys, os, numpy, re, psutil
import webbrowser as wb
from operator import itemgetter
from textwrap import wrap
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn import manifold
import scipy.spatial.distance as dist

#####
try:
    import igraph as ig
    from igraph.clustering import VertexClustering
    from igraph.layout import Layout
    from igraph import Edge
    import plotly
    import plotly.plotly as py
    import plotly.graph_objs as go
except ModuleNotFoundError as mne:
    print(mne)
    print("Skipping import for now")



from wombat_api.lib import *
TITLE_WRAP=40

def interpolate(start, end):
    mid =  (start+end)/2
    return([(start+mid)/2,mid,(mid+end)/2])

def draw_word_graph(matrix, xwords, ywords, layout="drl_3d", layout_params={'dim':3, 'weights':'weight'}, xstring="", ystring="", plot_name="", title="", verbose=False, minw=0, maxw=0.4, silent=True):
#    print(xwords)
#    print(ywords)
#    print(matrix)
    bipartite = layout.lower()=="bipartite"

    if plot_name=="":
        plot_name="graph-"+str(os.getpid())+".html"

    if bipartite:
        igraph = ig.Graph.Bipartite([False,True],[(0,1)], directed=False)
        igraph.delete_edges([(0,1)])
        igraph.delete_vertices([0,1])
    else:
        igraph = ig.Graph(directed=False)

    # Loop over data dimensions and create nodes
    for i in range(len(ywords)):
        node_attributes={'type':True, 'label':ywords[i]}
        igraph.add_vertex(name=str(i)+"_a_"+ywords[i], **node_attributes)
        for j in range(len(xwords)):
            if i == 0:
                node_attributes={'type':False, 'label':xwords[j]}
                igraph.add_vertex(name=str(j)+"_b_"+xwords[j], **node_attributes)
            #print(matrix[i,j])
            if matrix[i,j] >= minw and matrix[i,j] <= maxw  and numpy.isinf(matrix[i,j])==False:
                igraph.add_edge(str(i)+"_a_"+ywords[i],str(j)+"_b_"+xwords[j], **{'weight':int(matrix[i, j]), 'label':matrix[i, j]})

    layt=igraph.layout(layout, **layout_params)
    axes=dict(showbackground=False, showline=False, zeroline=False, showgrid=True, showticklabels=False, title='', showspikes=False)    
    if bipartite:
        scene=dict(xaxis=dict(axes), yaxis=dict(axes))
    else:
        scene=dict(xaxis=dict(axes), yaxis=dict(axes), zaxis=dict(axes))
    traces=[]

    # Create fake trace 
    if bipartite:
        traces.append(go.Scatter(x=[0], y=[0], mode='markers', marker={'color':'rgb(0, 0, 0)', 'opacity': 1, 'size': 0.1}, showlegend=False))
    else:
        traces.append(go.Scatter3d(x=[0], y=[0], z=[0], mode='markers', marker={'color':'rgb(0, 0, 0)', 'opacity': 1, 'size': 0.1}, showlegend=False))
    
    if bipartite:   
        word_node_trace=go.Scatter(x=[layt[igraph.vs[k].index][0] for k in range(len(igraph.vs))], 
                                 y=[layt[igraph.vs[k].index][1] for k in range(len(igraph.vs))],
                                mode='text', 
                                text=[igraph.vs[k]['label'] for k in range(len(igraph.vs))], 
                                textfont=dict(size=20),
                                hoverinfo='none',
                                #marker={'opacity': 1}
                                )
        traces.append(word_node_trace)
    else:
        word_node_trace1=go.Scatter3d(x=[layt[igraph.vs[k].index][0] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == True], 
                                 y=[layt[igraph.vs[k].index][1] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == True],
                                 z=[layt[igraph.vs[k].index][2] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == True],
                                mode='text', 
                                text=[igraph.vs[k]['label'] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == True], 
                                textfont=dict(size=20, color="red"),
                                hoverinfo='text',
                                marker={'opacity': 1})

        word_node_trace2=go.Scatter3d(x=[layt[igraph.vs[k].index][0] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == False], 
                                 y=[layt[igraph.vs[k].index][1] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == False],
                                 z=[layt[igraph.vs[k].index][2] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == False],
                                mode='text', 
                                text=[igraph.vs[k]['label'] for k in range(len(igraph.vs)) if igraph.vs[k]['type'] == False], 
                                textfont=dict(size=20, color="green"),
                                hoverinfo='text',
                                marker={'opacity': 1})

        traces.append(word_node_trace1)
        traces.append(word_node_trace2)

    # Edge start and end points
    Xs,Ys,Zs=[],[],[]
    # Edge label start and end points
    Xlabels,Ylabels,Zlabels=[],[],[]
    edge_labels=[]
    for e in igraph.es:
        Xs+=[layt[e.tuple[0]][0],layt[e.tuple[1]][0], None]
        Ys+=[layt[e.tuple[0]][1],layt[e.tuple[1]][1], None]
        if not bipartite: Zs+=[layt[e.tuple[0]][2],layt[e.tuple[1]][2], None] 

        for i in range (3):
            edge_labels.append(e['label'])

        # Get coords for nodes along the edge, for adding sim hover text to
        Xlabels.extend(interpolate(layt[e.tuple[0]][0],layt[e.tuple[1]][0]))
        Ylabels.extend(interpolate(layt[e.tuple[0]][1],layt[e.tuple[1]][1]))
        if not bipartite: 
            Zlabels.extend(interpolate(layt[e.tuple[0]][2],layt[e.tuple[1]][2]))

    # Draw edges
    if bipartite:
        traces.append(go.Scatter(x=Xs, y=Ys, mode='lines', opacity=1, line=dict(color='rgb(125,125,125)', width=2)))
    else:
        traces.append(go.Scatter3d(x=Xs, y=Ys, z=Zs, mode='lines', opacity=1,  line=dict(color='rgb(125,125,125)', width=1)))

    if bipartite:
        label_node_trace=go.Scatter(x=Xlabels, 
                                 y=Ylabels,
                                mode='markers', 
                                #text=[igraph.es[k]['label'] for k in range(len(igraph.es))], 
                                #textfont=dict(size=20),
                                hoverinfo='text',
                                hovertext=edge_labels)
    else:
        label_node_trace=go.Scatter3d(x=Xlabels, 
                                 y=Ylabels,
                                 z=Zlabels, 
                                mode='markers', 
                                #text=[igraph.es[k]['label'] for k in range(len(igraph.es))], 
                                #textfont=dict(size=20),
                                hoverinfo='text')
    traces.append(label_node_trace)


    layout = go.Layout(
             title=xstring+" <--> "+ystring,
             width=1800,
             height=1000,
             showlegend=False,
             scene=scene,
             margin=dict(t=100),
             angularaxis=dict(visible=True),
             hovermode='closest', 
             hoverdistance=-1,
             xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
             yaxis=dict(showgrid=False, zeroline=False, showticklabels=False))
    fig=go.Figure(data=traces, layout=layout)
    plotly.offline.plot(fig, filename=plot_name, auto_open=(silent==False))


def plot_pairwise_distances(vector_result1, vector_result2, pdf_name="", size=(10,10), share_axes=('all','none'), max_pairs=20,  verbose=False, fontsize=14, metric=dist.cosine, arrange_by="", silent=False, ignore_identical=False, bar_props={'color':'lightgreen'}, text_props={'fontsize':14}, axis_props={'labelsize':14}):
    
    cartesian_mode=False

    if vector_result2 == None:
        print("Creating cartesian product of single input list!")
        cartesian_mode=True

    if not cartesian_mode:
        # Each input must contain data from the same number of WECs
        assert len(vector_result1) == len(vector_result2)

    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(arrange_by)
    
    pdf_name=pdf_name if pdf_name!="" else str(psutil.Process().pid)+"_plot.pdf"
    pdf=PdfPages(pdf_name)

    # Iterate over pages (there is at least one)
    for pages in range(plot_pages):
        items_on_page={}
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
            axes[row,col].tick_params(**axis_props)
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
                    if ignore_identical == False or not np.array_equal(s1_avg,s2_avg):
                        results.append((t, f1[t][0], f2[t][0], float(metric(s1_avg, s2_avg))))
            else:
                # f1 is the complete result for WEC p
                f1=vector_result1[p][1]
                temp,results=[],[]
                for t in range(len(f1)):
                    vecs1=[]
                    for v in f1[t][2]:
                        vecs1.append(v[1])
                    s1_avg = np.average(vecs1, axis=0)
                    temp.append((s1_avg, f1[t][0]))
                c=0
                for outer in range(len(temp)):
                    s1_avg=temp[outer][0]
                    sent1=temp[outer][1]
                    for inner in range(outer+1,len(temp)):
                        s2_avg=temp[inner][0]
                        sent2=temp[inner][1]
                        c+=1
                        if ignore_identical == False or not np.array_equal(s1_avg,s2_avg):
                            results.append((c, sent1, sent2, float(metric(s1_avg, s2_avg))))

            # Sort by distance 
            data=sorted(results, key=itemgetter(3), reverse=False)
            data=data[:max_pairs]
            pair_ids, distances, distance_labels, sentences=[],[],[],[]
            pair_pos = np.arange(max_pairs)
            for d in data:                
                pair_ids.append("{:03d}".format(d[0])+" ({0:0.5f})".format(d[3]))
                try:
                    items_on_page[d[0]]+=1
                except KeyError:
                    items_on_page[d[0]]=1
                distances.append(d[3])
                sentences.append(d[1]+" <--> "+d[2])                                
            max_dist = max(max_dist,float(distances[-1]))
            axes[row,col].barh(pair_pos,distances,**bar_props)
            axes[row,col].set_yticklabels(pair_ids)
            # Add bars 
            for rect, sentence in zip(axes[row,col].patches, sentences):
                axes[row,col].text(rect.get_x()+0.0001, rect.get_y()+0.4, sentence, ha='left', va='center', **text_props)

        plt.xlim(0,max_dist*1.1)
        pdf.savefig()
    pdf.close()
    if not silent: wb.open(pdf_name)



def plot_tsne(vector_result, pdf_name="", iters=250, size=(10,10), share_axes=('none','none'), x_lim=None, y_lim=None, fontsize=14, arrange_by="", highlight="", suppress="", silent=True):

    (we_params_dict_list, plot_coords, plot_rows, plot_cols, plot_pages) = expand_parameter_grids(arrange_by)
    
    pdf_name=pdf_name if pdf_name!="" else str(psutil.Process().pid)+"_plot.pdf"
    pdf=PdfPages(pdf_name)
    x_to_share=share_axes[0]
    y_to_share=share_axes[1]

    if x_lim!=None:
        x_to_share='all'
    if y_lim!=None:
        y_to_share='all'

    # Iterate over pages (there is at least one)
    for pages in range(plot_pages):
        # Create container for current page
        fig,axes = plt.subplots(nrows=plot_rows, ncols=plot_cols, 
                        figsize=(size[0]*plot_cols, size[1]*plot_rows), 
                        squeeze=False, sharex=x_to_share, sharey=y_to_share)        

        for p,we_params_dict in enumerate(we_params_dict_list):
            (row,col,page)=plot_coords[p]
            if pages != page: continue
            vectors,units=[],[]
            for (_, _, tuples) in vector_result[p][1]:
                for (w,v) in tuples:
                    units.append(w)
                    vectors.append(v)
            x = np.asarray(vectors)
#            print(x)
            sys.stdout.write("Doing tsne magic ...")
            sys.stdout.flush()
            tsne = manifold.TSNE(n_components=2, init='random', random_state=9, n_iter=iters)
            #tsne = manifold.TSNE(n_components=2, init='pca', n_iter=iters)
#            print(x.shape)
#            init=np.ones((len(vectors),2))
#            init.fill(.1)
#            print(init)
            #tsne = manifold.TSNE(n_components=2, init=init, n_iter=iters, n_iter_without_progress=1000, learning_rate=500)
            print(x)
            print(tsne.get_params())
            y = tsne.fit_transform(x)
            print(y)
            sys.stdout.write(" done\n")
            sys.stdout.flush()

            if x_lim !=None:
                axes[row,col].set_xlim(x_lim)
            if y_lim !=None:
                axes[row,col].set_ylim(y_lim)

            axes[row,col].scatter(y[:, 0], y[:, 1], marker='.', c='w')
            axes[row,col].set_title(vector_result[p][0])
            for i, txt in enumerate(units):
                if suppress != "" and re.match(suppress, txt)!=None:
                    continue
                if highlight=="":
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='black', fontweight='normal')
                elif re.match( highlight, txt)!=None:
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='green', fontweight='bold')
                else:
                    axes[row,col].annotate(txt, (y[:,0][i], y[:,1][i]), fontsize=fontsize, color='black', fontweight='normal')
        # end page

        pdf.savefig()    
    pdf.close()
    if not silent: wb.open(pdf_name)


def compute_unit_distance_matrices(vector_result1, vector_result2, metric=dist.cosine, ignore_matching=True, invert=False, ignore=['*sw*']):
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
            ignorable = list(set(ignorable))

            to_del1 = [i for i,tup in enumerate(tuples1) if tup[0] in ignorable]
            to_del2 = [i for i,tup in enumerate(tuples2) if tup[0] in ignorable]

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
                if word2 not in ywords:
                    ywords.append(word2)
                for colnum in range(len(tuples1)):
                    (word1,vector1) = tuples1[colnum]
                    if numpy.isnan(vector1).any() or word1 in ignorable:
                        continue
                    if len(xwords) < len(tuples1):#-len(ignorable)):
                        if word1 not in xwords:
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


""" 
Return from the word embeddings specified by 'we_param_grid_string' the 'count' (default 10) items 
most similar to each word in 'targets', from most to least similar (=least to most distant). 
Use two-place function 'measure' (default scipy.spatial.distance.cosine) for computing the similarity.
If 'to_rank' is a list of strings, only their similarity to 'target' is 
computed and returned, sorted from most to least similar.
Returns a list of <result, target, we_desc> tuples, where result is itself a list of <word, sim> tuples.
"""
def get_most_similar(wb, we_param_grid_string, targets=[], count=10, measure=dist.cosine, to_rank=[], verbose=False):
    (we_params_dict_list,_ ,_ ,_ ,_) = expand_parameter_grids(we_param_grid_string)
    all_results=[]
    # Iterate over all wecs in the outer loop, and read each only once
    for we_params_dict in we_params_dict_list:
        current_emb=None
        if len(to_rank)>0:
            #retrieved=wb.get_vectors(for_input=[to_rank], as_tuple=True, verbose=verbose)
            pass
            # TODO
        else:
            current_emb = wb.get_all_vectors(we_params_dict,as_tuple=True)[0][1][0][2]
            #retrieved=retrieved[0][1][0][2]

        # Iterate over the target words
        for target in targets:
            target_tuple=wb.get_vectors(we_params_dict, {}, for_input=[[target]])[0][1][0][2][0]
            #target_tuple=target_tuple[0][1][0][2][0]
            if np.isnan(target_tuple[1][0]):
                print("Target '%s' not found in '%s'"%(target,dict_to_sorted_string(we_params_dict, pretty=True)))
                continue


            current_dist=float(0.0)
            result = []
            for row in current_emb:
                # Each row is a flat (w,v) tuple
                if row[0] == target: continue
                current_dist = float(measure(target_tuple[1], row[1]))
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
            all_results.append((result,target,dict_to_sorted_string(we_params_dict,pretty=True)))
    return all_results


