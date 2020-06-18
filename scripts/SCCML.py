import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import subplots,scatter
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.externals import joblib
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
import scipy.stats as stats
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
#from yellowbrick.features import RFECV
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
import xgboost
import shap
from adjustText import adjust_text


def plot_featureImp_glob(model,features,feature_num_shown):
    
    feature_imp = model.feature_importances_
#     feature_imp_index_ranked = np.argsort(feature_imp)[::-1]
#     features_ranked = features[feature_imp_index_ranked]
    feature_imp = feature_imp/feature_imp.max()

    fig,ax = subplots(figsize=(6,4))
    ax.stem(features,feature_imp,markerfmt=' ')
    
    features_selected = []
    for i in range(feature_num_shown):

        ax.scatter(float(features_ranked[i]),feature_imp[feature_imp_index_ranked][i],color='k',s=10)

        ax.annotate(format(float(features_ranked[i]),'.4f'), xy=(float(features_ranked[i]),
            feature_imp[feature_imp_index_ranked][i]),fontsize=8)

        features_selected.append(float(features_ranked[i]))

    ax.set_ylabel('relative importance')
    ax.set_xlabel('m/z')

    return fig, features_selected

def feature_contrib(model,X,features,feature_num_shown,if_summary):
    
    shap_explainer = shap.TreeExplainer(model)
    shap_vals = shap_explainer.shap_values(X)
    
    fig1,axes = subplots(figsize=(12,4))
    
    g=axes.stem([float(x) for x in features],shap_vals.mean(axis=0),markerfmt=' ',linefmt='k')
    axes.get_yaxis().set_ticks([])
    shap_vals_index_ranked = np.argsort(shap_vals.mean(axis=0))[::-1]
    shap_vals_ranked = shap_vals.mean(axis=0)[shap_vals_index_ranked]
    #axes.spines['right'].set_visible(False)
    #axes.spines['top'].set_visible(False)
    texts = []
    axes.scatter(features[shap_vals_index_ranked[:feature_num_shown]],shap_vals_ranked[:feature_num_shown],color='b',s=35,marker='v')
    axes.scatter(features[shap_vals_index_ranked[-feature_num_shown:]],shap_vals_ranked[-feature_num_shown:],color='orange',s=35,marker='s')
    for i in range(feature_num_shown):
        texts.append(plt.text(float(features[shap_vals_index_ranked[i]]),shap_vals_ranked[i],float(features[shap_vals_index_ranked[i]]),fontsize=12))
        texts.append(plt.text(float(features[shap_vals_index_ranked[-i-1]]),shap_vals_ranked[-i-1],float(features[shap_vals_index_ranked[-i-1]]),fontsize=12))

#         axes.annotate(format(float(features[shap_vals_index_ranked[i]]),'.2f'), xy=(float(features[shap_vals_index_ranked[i]]),shap_vals_ranked[i]),fontsize=10)
#         axes.annotate(format(float(features[shap_vals_index_ranked[-i-1]]),'.2f'), xy=(float(features[shap_vals_index_ranked[-i-1]]),shap_vals_ranked[-i]),fontsize=10)
    axes.set_xlabel('m/z',fontsize=12)
    axes.set_ylabel('mean SHAP values',fontsize=12)
    axes.legend()
    adjust_text(texts)
    if if_summary:
        fig2,axes = subplots(figsize=(12,4))
        shap.summary_plot(shap_vals,X,max_display=15)
    return fig1,fig2,shap_vals

from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import MaxNLocator


def shap_clustering(shap_vals,groups):
    
    shap_pca50 = PCA(n_components=12).fit_transform(shap_vals)
    #shap_embedded = TSNE(n_components=2, perplexity=25).fit_transform(shap_vals)
    
    cdict1 = {
        'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
                (1.0, 0.9607843137254902, 0.9607843137254902)),

        'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
                  (1.0, 0.15294117647058825, 0.15294117647058825)),

        'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
                 (1.0, 0.3411764705882353, 0.3411764705882353)),

        'alpha': ((0.0, 1, 1),
                  (0.5, 1, 1),
                  (1.0, 1, 1))
    }  # #1E88E5 -> #ff0052
    red_blue_solid = LinearSegmentedColormap('RedBlue', cdict1)
    
    shap_embedded_df = pd.DataFrame(shap_pca50)
    shap_embedded_df['type'] = groups

    f1,axes= subplots(1,1,figsize=(5,4))
    g = sns.scatterplot(x=0,y=1,hue='type',data=shap_embedded_df,ax=axes)
    g.legend(loc='upper right')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.get_yaxis().set_ticks([])
    axes.get_xaxis().set_ticks([])
    axes.set_xlabel('PC1',fontsize=12)
    axes.set_ylabel('PC2',fontsize=12)
    
    f2,axes= subplots(1,1,figsize=(6,4))
    plt.scatter(shap_pca50[:,0],
           shap_pca50[:,1],
           c=shap_vals.sum(1).astype(np.float64),
           linewidth=0, alpha=0.9, cmap=red_blue_solid)
    cb = plt.colorbar(label="Log odds of being hippocampal or cerebellar", aspect=40, orientation="vertical")
    cb.set_alpha(1)
    cb.draw_all()
    cb.outline.set_linewidth(0)
    cb.ax.tick_params('x', length=0)
    cb.ax.xaxis.set_ticks_position("top")
    cb.ax.xaxis.set_label_position('bottom')
    axes.spines['right'].set_visible(False)
    axes.spines['top'].set_visible(False)
    axes.get_yaxis().set_ticks([])
    axes.get_xaxis().set_ticks([])
    axes.set_xlabel('PC1',fontsize=12)
    axes.set_ylabel('PC2',fontsize=12)
    plt.show()
    return f1,f2,shap_pca50

def plot_shap_features(shap_vals,X,FOI):
    
    shap_pca = PCA(n_components=12).fit_transform(shap_vals)
    cdict1 = {
    'red': ((0.0, 0.11764705882352941, 0.11764705882352941),
            (1.0, 0.9607843137254902, 0.9607843137254902)),

    'green': ((0.0, 0.5333333333333333, 0.5333333333333333),
              (1.0, 0.15294117647058825, 0.15294117647058825)),

    'blue': ((0.0, 0.8980392156862745, 0.8980392156862745),
             (1.0, 0.3411764705882353, 0.3411764705882353)),

    'alpha': ((0.0, 1, 1),
              (0.5, 1, 1),
              (1.0, 1, 1))
    }  # #1E88E5 -> #ff0052
    red_blue_solid = LinearSegmentedColormap('RedBlue', cdict1)
    row_plot = int(np.ceil(len(FOI)/3))
    f,axes = subplots(3,row_plot,figsize=(row_plot*5,10))
    ax = axes.ravel()
    index = 0
    for feature in FOI:
        fig=ax[index].scatter(shap_pca[:,0],
                   shap_pca[:,1],
                   c=X[feature].values[:10000].astype(np.float64),
                   linewidth=0, alpha=0.75, cmap=red_blue_solid)
        cb=plt.colorbar(fig, aspect=40, orientation="vertical",ax=ax[index])
        cb.set_alpha(1)
        cb.set_label(label='normalized intensity',size=10)
        cb.draw_all()
        cb.outline.set_linewidth(0)
        cb.ax.tick_params('x', length=0)
        cb.ax.xaxis.set_label_position('top')
        ax[index].set_title(format(float(feature),'.4f')+' m/z',fontsize=12)
        ax[index].axis("off")
        index += 1

    return f