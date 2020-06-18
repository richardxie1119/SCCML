#from scipy.stats import wilcoxon
from scipy.stats import ranksums
#from scipy.stats import mannwhitneyu
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import louvain
import igraph as ig
import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix
from sklearn.manifold import TSNE
import umap
import re
from statsmodels import robust
from pyteomics import mzml
from matplotlib.pyplot import subplots,scatter
import matplotlib.patches as mpatches
import seaborn as sns
import requests
import io

def readmzXML(file_paths):
    
    data_dict = {}
    for file_path in file_paths:
        data = mzml.MzML(file_path)
        file_num = len(data)

        for spectrum in data:
            data_dict[spectrum['spectrum title']] = {'m/z':spectrum['m/z array'],'intensity':spectrum['intensity array']}

        print('successfully loaded '+str(file_num)+' raw spectra from'+file_path)

    return data_dict

def extracSampleID(sample_string_list):

    sampleID = []
    for sample_string in sample_string_list:

        x_loc = re.search('x_(.*)y_',sample_string).group(1)
        #print(x_loc)
        y_loc = re.search(r'y_(.*)\\1',sample_string).group(1).split('\\')[0]
        #print(y_loc)
        ID = 'x_'+x_loc+'y_'+y_loc

        sampleID.append(ID)


    return sampleID

def PCanalysis(num_com, num_plot, data, if_plot):
    pca = PCA(n_components=num_com)
    pca_results = pca.fit_transform(data)
    if if_plot == True:
        fig, axes = plt.subplots(1, num_plot, figsize=(15, 4), sharex=True, sharey=True)
        ax = axes.ravel()
        #     for i in range(num_com):
        #         print('Explained variation per principal component: {}'.format(pca.explained_variance_ratio_[i]))
        for j in range(num_plot):
            ax[j].scatter(pca_results[:, j], pca_results[:, j + 1])
            ax[j].set_title('Explained variation {}'.format(round(pca.explained_variance_ratio_[j], 3)))
        # fig.show()
    return pca_results



def rank_sum_test(x, y, features, N):
    dim = x.shape[1]
    sample_num = min(x.shape[0], y.shape[0])
    S = []
    P = []
    for i in range(dim):
        s, p = ranksums(x[:, i], y[:, i])
        S.append(s)
        P.append(p)

    S = np.asarray(S)
    P = np.asarray(P)

    P_order = P.argsort()[-N:]
    # P_ranks = P_order.argsort()
    feature_order = features[P_order]

    return P_order, feature_order, P

def TSNE_(n_comp,perpl,n_iter,random_state,data):
    
    tsne_list = []
    for i in perpl:
        for j in n_iter:
            tsne_dict = {}

            print('TSNE in progress with parameters: {}'.format([n_comp,i,j]))
            
            tsne = TSNE(n_components=n_comp, perplexity=i, n_iter=j,random_state=random_state)          
            tsne_results = tsne.fit_transform(data)
            
            tsne_dict['result'] = tsne_results
            tsne_dict['parameters'] = [i,j]
        tsne_list.append(tsne_dict)
        
    return tsne_list

def UMAP_(n_comp,min_dist_list,n_neigh_list,metric,data):
    
    umap_list = []
    for min_dist in min_dist_list:
        for n_neigh in n_neigh_list:
    
            print('UMAP in progress with parameters: {}'.format([n_comp,min_dist,n_neigh,metric]))
        
            umap_dict = {}
            
            fit = umap.UMAP(n_components=n_comp,n_neighbors=n_neigh, min_dist=min_dist,metric=metric)
            umap_results = fit.fit_transform(data)
            umap_dict.update({'result':umap_results})
            umap_list.append(umap_dict)
    
    return umap_list
# construct adjacency matrix based on nearest neighbors search with jaccard metric

def construct_graph_KNN(data,n_neigh,n_jobs,if_jaccard):
    if if_jaccard:
        method = 'jaccard'
    else:
        method = 'euclidean'

    print('Now building KNN graph with the {}'.format(method)+' metric with {}'.format(n_neigh)+' of neighbors...')
    
    nn_jaccard = NearestNeighbors(n_neighbors=n_neigh, metric=method,n_jobs=n_jobs)
    nn_jaccard.fit(data)
    kneigh_dist,kneigh_idx = nn_jaccard.kneighbors()
    kneigh_graph = nn_jaccard.kneighbors_graph()
    #kneigh_graph_array = nn_jaccard.kneighbors_graph()

    print(kneigh_graph.shape)

    
    # if if_jaccard:

    #     dm_count = 0
        
    #     # Compute condensed distance matrix (upper triangle) of pairwise dtw distances
    #     # when x and y are the same array
    #     x_s = kneigh_graph.shape[0]
    #     dm = lil_matrix((x_s * (x_s - 1)) // 2)

    #     for i in range(0, x_s[0] - 1):
    #         for j in range(i + 1, x_s[0]):

    #             dm[i,j] = len(set(kneigh_idx[i]).intersection(kneigh_idx[j]))/len(set(kneigh_idx[i]).union(kneigh_idx[j]))
    #             dm_count += 1

    #     # Convert to squareform
    #     sources, targets = dm.nonzero()
    #     dm[targets,sources] = dm[sources,targets]



    dist_list = []
    for i in kneigh_dist:
        dist_list.extend(i)

    vcount = max(kneigh_graph.shape)
    sources, targets = kneigh_graph.nonzero()
    edgelist = list(zip(sources.tolist(), targets.tolist()))
    g = ig.Graph(vcount,edgelist)
    g.vs['label'] = data.index.values.tolist()
    g.es['weight'] = dist_list
    
    print('successful')
    
    return g


def louvain_clus(graph):
    
    partition = louvain.find_partition(graph, louvain.ModularityVertexPartition)
    
    print(partition.summary())
    
    subgraphs = partition.subgraphs()
    subgraph_labels_df = pd.DataFrame(columns=['label','cluster'])
    index = 0
    for i in range(len(subgraphs)):
        subgraph_labels = subgraphs[i].vs['label']
        for label in subgraph_labels:
            subgraph_labels_df.loc[index] = [label,i]
            index = index + 1 
    print('Done')
    
    return partition,subgraph_labels_df

def q_n(a):
    """Rousseeuw & Croux's (1993) Q_n, an alternative to MAD.

    ``Qn := Cn first quartile of (|x_i - x_j|: i < j)``

    where Cn is a constant depending on n.

    Finite-sample correction factors must be used to calibrate the
    scale of Qn for small-to-medium-sized samples.

        n   E[Qn]
        --  -----
        10  1.392
        20  1.193
        40  1.093
        60  1.064
        80  1.048
        100 1.038
        200 1.019

    """
    a = np.asarray(a)

    # First quartile of: (|x_i - x_j|: i < j)
    vals = []
    for i, x_i in enumerate(a):
        for x_j in a[i+1:]:
            vals.append(abs(x_i - x_j))
    quartile = np.percentile(vals, 25)

    # Cn: a scaling factor determined by sample size
    n = len(a)
    if n <= 10:
        # ENH: warn when extrapolating beyond the data
        # ENH: simulate for values up to 10
        #   (unless the equation below is reliable)
        scale = 1.392
    elif 10 < n < 400:
        # I fitted the simulated values (above) to a power function in Excel:
        #   f(x) = 1.0 + 3.9559 * x ^ -1.0086
        # This should be OK for interpolation. (Does it apply generally?)
        scale = 1.0 + (4 / n)
    else:
        scale = 1.0

    return quartile / scale

from scipy import signal

def screenSpectra(data,sample_rate,estimator,lambda_,if_plot):
    

    score = []
    for index,spectra in data.iterrows():
        intens = spectra.values
        mass = spectra.index.values

        spec_resampled = signal.resample(intens,sample_rate,mass)
        median_intens = np.median(spec_resampled[0])
        #print(median_intens)
        if estimator == 'mad':
            S = robust.mad(spec_resampled[0])
            #print(S)
            A = (S**lambda_)/((median_intens+1)**((1-lambda_)/2))
        if estimator == 'Q':
            Q = q_n(spec_resampled[0])
            print(Q)
            
            A = (Q**lambda_)/((median_intens+1)**((1-lambda_)/2))
            
        score.append(A)
    score = np.asarray(score)
    
    if if_plot:
        fig, ax = plt.subplots()
        ax.scatter(np.linspace(1,len(score),len(score)),score)
        
    return score

import networkx as nx

def build_MAN(classes,classes_pos,masses,weights):
    
    G = nx.Graph()
    #G.add_nodes_from([1,2])
    edge_list = []
    
    for i in range(len(classes)):
        class_ = classes[i]
        
        for j in range(len(masses)):
            
            edge_list.append((class_,masses[j],{'weight':weights[i][j]}))
            
    G.add_edges_from(edge_list)
        
    for i in range(len(classes)):
        G.node[classes[i]]['pos'] = classes_pos[i]
        
    return G, edge_list

def LipidMaps_annotate(mass_list,adducts,tolerance,site_url):
    
    Data = []
    matched = []
    unmatched = []
    
    for mass in mass_list:
        Data_ = []
        for adduct in adducts:
            url = site_url+'/{}/{}/{}'.format(mass,adduct,tolerance)
            
            urlData = requests.get(url).content.decode('utf-8')[7:-9]            
            rawData = pd.read_csv(io.StringIO(urlData),sep='\t',error_bad_lines=False,index_col=False)
            
            Data_.append(rawData)
            Data.append(rawData)
        df = pd.concat(Data_, ignore_index=True)
        
        if df.empty:
            unmatched.append(mass)
        else:
            matched.append(mass) 
            
    annot_df = pd.concat(Data, ignore_index=True)
    return annot_df, matched, unmatched

    
def find_markers(data,tsne_df,clusters,num_top_feature):
    
    feature_list = np.asarray(data.columns)
    
    markers_df_final = pd.DataFrame()
    
    for cluster in clusters:
        cluster_data = data.loc[tsne_df['label'][tsne_df['cluster']==cluster].values].values
        
        test_data = data.drop([tsne_df['label'][tsne_df['cluster']==cluster].values][0],axis=0).values
        
        O, F, P = analysis.rank_sum_test(cluster_data,test_data, feature_list, 10000)
    
        markers_df = pd.DataFrame()
        markers_df['markers'] = F[:num_top_feature]
        markers_df['p values'] = P[O][:num_top_feature]
        markers_df['cluster'] = cluster
        
        markers_df_final = pd.concat([markers_df_final, markers_df])
        
    markers_df_final = markers_df_final.reset_index(drop=True)
    
    return markers_df_final

def plot_markers(data,tsne_df,clusters,num_top_feature):
    
    markers_df = find_markers(data,tsne_df,clusters, num_top_feature)
    
    fig, axes = plt.subplots(6,6)
    ax = axes.ravel()
    index = 0
    
    for cluster_name in cluster_names:
        for i in range(num_top_feature):
            marker_name = markers_df['markers'][markers_df['cluster']==cluster_name].values[i]
            ax[index].scatter(x=tsne_df['tsne_X'].values, y=tsne_df['tsne_Y'].values, 
                       c=data_clust[marker_name].values.flatten(),
                       s=20, edgecolor='', alpha=0.8,cmap='YlOrRd')
            ax[index].set_title(str(marker_name)+ ' m/z')
            ax[index].set_yticks([])
            ax[index].set_xticks([])
            index = index+1
            
    return markers_df

def plotSpectraFeature(data,mass_list,data_info,line_list,features):
    
    types_list = list(set(data_info['type']))

    for feature in features:
        
        fig, axes = subplots(1,2)
        ax = axes.ravel()
        mass_range = [np.where(mass_list==float(feature))[0][0]-20,np.where(mass_list==float(feature))[0][0]+30]
        handles = []
        for t in range(len(types_list)):
            data_ = data[data_info['type']==types_list[t]]
            for i in range(data_.shape[0]):
                ax[0].plot(mass_list[mass_range[0]:mass_range[1]],data_[i,mass_range[0]:mass_range[1]],line_list[t],alpha=0.4,linewidth=0.5)
                
                
            handel = mpatches.Patch(linewidth=1,color=line_list[t][2:],label=types_list[t])
            handles.append(handel)
            ax[0].legend(handles=handles,fontsize=9)
                
            sns.distplot(data_[:,np.where(mass_list==float(feature))[0][0]],bins=40,ax=ax[1],label=types_list[t])
            ax[1].legend()
        fig.suptitle(str(feature)+' m/z')

def mass_match(data,mass_df,adducts_list,tolerance):
    
    matched_data = pd.DataFrame()
    mass_data = pd.DataFrame()
    mass_data2 = pd.DataFrame()
    matched_mass = []
    matched_species_unique = []
    
    if 'cluster' in data.columns:
        matched_data.insert(0,'cluster',data['cluster'].values)
        data = data.drop('cluster',axis=1)
    
    if 'type' in data.columns:
        matched_data.insert(0,'type',data['type'].values)
        data = data.drop('type',axis=1)
        
    if 'animal' in data.columns:
        matched_data.insert(0,'animal',data['animal'].values)
        data = data.drop('animal',axis=1)

    if 'batch' in data.columns:
        matched_data.insert(0,'batch',data['batch'].values)
        data = data.drop('batch',axis=1)
        
    data_mass = data.columns.values.astype('float')

    for Species in mass_df.iterrows():
        Species = Species[1]
        species_name = Species['Compound Name']
        platform = 'Primary'
        
        for adducts in adducts_list:
            accurate_mass = Species[adducts]
            column_ = (10**6*abs(accurate_mass-data_mass)/accurate_mass) < tolerance
            if np.where(column_==True)[0].size > 0:
                
                data_selected = data.iloc[:,column_]
                min_error_mass = np.argmin(abs(data_selected.columns.values-accurate_mass))
                min_mass = data_selected.columns.values[min_error_mass]
                data_selected_ = data_selected.iloc[:,min_error_mass]
                
                column_sum = data_selected_.sum()
                matched_data[float(min_mass)] = data_selected_.values
                cells_presence = data_selected_.astype(bool).sum(axis=0)
                
                if cells_presence > 0:
                    if str(species_name)+'_'+str(adducts) in matched_data.columns:
                        #matched_data[str(species_name)+'_'+str(adducts)+'_d'] = data_selected_.values
                        mass_data[str(species_name)+'_'+str(adducts)+'_d'] = [str(species_name),accurate_mass,
                                                                              data_selected.columns.values[min_error_mass],
                                                                              10**6*abs(min_mass-accurate_mass)/accurate_mass,
                                                                              cells_presence,
                                                                             platform]

                    else:
                        #matched_data[str(species_name)+'_'+str(adducts)] = data_selected_.values
                        mass_data[str(species_name)+'_'+str(adducts)] = [str(species_name),accurate_mass,
                                                                          data_selected.columns.values[min_error_mass],
                                                                          10**6*abs(min_mass-accurate_mass)/accurate_mass,
                                                                          cells_presence,
                                                                        platform]           
                
    
    mass_data.set_index([pd.Series(['species','accurate_mass','measured mass','ppm_error','cells_presence','platform'])],inplace=True)
    mass_data = mass_data.T
    
    mass_data2 = pd.DataFrame()
    for mass in set(mass_data['accurate_mass']):
        species_list = list(mass_data[mass_data['accurate_mass']==mass].index.values)
        accurate_mass = mass_data[mass_data['accurate_mass']==mass]['measured mass'].values[0]
        cells_presence = mass_data[mass_data['accurate_mass']==mass]['cells_presence'].values[0]
        platform = mass_data[mass_data['accurate_mass']==mass]['platform'].values[0]
        mass_data2[mass] = [species_list,
                            accurate_mass,
                            10**6*abs(mass-accurate_mass)/accurate_mass,
                            cells_presence,
                            platform]
        mass_data2.set_index([pd.Series(['species_list','measured_mass','ppm_error','cells_presence','platform'])],inplace=True)
    mass_data2 = mass_data2.T
    
    return matched_data,mass_data,mass_data2

def clustering_main(data,metric):

    data_out = data.copy()

    if 'type' in data.columns:
        data = data.drop('type',axis=1)
    if 'batch' in data.columns:
        data = data.drop('batch',axis=1)
    if 'animal' in data.columns:
        data = data.drop('animal',axis=1)
    if 'cluster' in data.columns:
        data = data.drop('cluster',axis=1)
        
    if metric == 'jaccard':
        data = data.astype(bool)
    else:
        data = data
    
    
    #kmeans = KMeans(n_clusters=2, random_state=19,max_iter=3000).fit_predict(data)
    #hierac = AgglomerativeClustering(n_clusters=2, affinity='euclidean', linkage='ward').fit_perdict(data_used) 
    tsne = TSNE(n_components=2, perplexity=30, n_iter=4000,random_state=19,metric=metric)
    tsne_results = tsne.fit_transform(data)
    graph = construct_graph_KNN(data,20,8,metric)
    partition,louvain_df = louvain_clus(graph)

    tsne_df = pd.DataFrame()
    tsne_df['tsne_X'] = tsne_results[:,0]
    tsne_df['tsne_Y'] = tsne_results[:,1]
    tsne_df['type'] = types
    tsne_df['animal'] = animal
    tsne_df['batch'] = batch
    tsne_df['label'] = data.index.values
    #tsne_df['cluster_kmeans'] = kmean
    #tsne_df['cluster_hierac'] = hierac
    tsne_df['cluster'] = ['null']*len(tsne_df['label'])

    for index, row in louvain_df.iterrows():
        tsne_df.loc[tsne_df['label'] == row['label'], 'cluster'] = row['cluster']
    
    data_out['cluster'] = tsne_df['cluster'].values
   
    return tsne_df,data_out