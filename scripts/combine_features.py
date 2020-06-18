# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 23:46:33 2019

@author: Richard
"""

O, F, P = analysis.rank_sum_test(intens_mat_1_pp,intens_mat_2_pp, masses_list_pp, 10000)

rf_best = joblib.load('rf_classifier_pp.sav')
y_pred = rf_best.predict(X_test_pp)
report_dict = classification_report(y_test_pp, y_pred, output_dict=True)
feature_importance = rf_best.feature_importances_
feature_index_ranked = np.argsort(feature_importance)[::-1]
feature_ranking = masses_list_pp[feature_index_ranked]

O_combined = np.concatenate((O[:59],feature_index_ranked[:87]),axis=0)
O_combined_unique = np.unique(O_combined)

#%%
X_train_selected, X_test_selected = X_train_pp[:, feature_index_ranked[:90]], X_test_pp[:, feature_index_ranked[:90]]

rfe_classifier = test_features.SVM_classi(X_train_selected,y_train_pp,'linear',param_grid,True,6)
fpr_rfe, tpr_rfe , roc_auc_rfe = test_features.roc(rfe_classifier,X_test_selected,y_test_pp,2)

scores = cross_val_score(rfe_classifier, data_pp[feature_index_ranked[:90]], labels2, cv=50,n_jobs=8)
scores
scores.mean(), scores.std()**2
#%%
X_train_selected, X_test_selected = X_train_pp[:, O[:72]], X_test_pp[:, O[:72]]

rfe_classifier = test_features.SVM_classi(X_train_selected,y_train_pp,'linear',param_grid,True,6)
fpr_rfe, tpr_rfe , roc_auc_rfe = test_features.roc(rfe_classifier,X_test_selected,y_test_pp,2)

scores = cross_val_score(rfe_classifier, data_pp[O[:90]], labels2, cv=50,n_jobs=8)
scores
scores.mean(), scores.std()**2

#%%
X_train_RS, X_test_RS = X_train_pp[:, O_combined_unique], X_test_pp[:, O_combined_unique]

Cs=[10**-6,10**-5,10**-4,10**-3,10**-2,0.1,10,10**2,10**3,10**4,10**5]
gammas=[10**-6,10**-5,10**-4,10**-3,10**-2,0.1,10,10**2,10**3,10**4,10**5]
param_grid = {'C': Cs, 'gamma': gammas}

grid_search = GridSearchCV(SVC(kernel='linear'), param_grid, cv=10, n_jobs=6)

grid_search.fit(X_train_RS, y_train_pp)
best_params = grid_search.best_params_
print('starting RFECV...')
# Create RFECV visualizer with linear SVM classifier
viz = RFECV(SVC(kernel='linear', C=best_params['C'], gamma=best_params['gamma']),n_jobs=6,cv=10,scoring='accuracy',step=1)
viz.fit(X_train_RS, y_train_pp)
viz.poof()

#%%
O_selected = O_combined_unique[viz.ranking_==1]
F_selected = O_combined_unique[viz.ranking_==1]

np.savetxt('O_selected_pp_combined_2.csv',[O_selected],delimiter=',')
#O_selected = pd.read_csv('O_selected_pp_combined_2.csv',header=None).astype(int).values[0,:]
X_train_selected, X_test_selected = X_train_pp[:, O_selected], X_test_pp[:, O_selected]

rfe_classifier = test_features.SVM_classi(X_train_selected,y_train_pp,'linear',param_grid,True,6)
fpr_rfe, tpr_rfe , roc_auc_rfe = test_features.roc(rfe_classifier,X_test_selected,y_test_pp,2)

scores = cross_val_score(rfe_classifier, data_pp[O_selected], labels2, cv=50,n_jobs=8)
scores
scores.mean(), scores.std()**2
#%%
sns.set_style("ticks")
log10_feature_importance = np.log10(feature_importance)
log10_feature_importance_ = log10_feature_importance[log10_feature_importance!=float('-inf')]
ax = sns.jointplot(np.log10(P)[log10_feature_importance!=float('-inf')],log10_feature_importance_,color='c',alpha=0.6)
for O_ in O_selected:
    ax.ax_joint.scatter(np.log10(P)[O_],log10_feature_importance[O_],color='m',alpha=0.6)
    
patch = mpatches.Patch(linewidth=1,color='m',label='selected featurers')
ax.ax_joint.legend(handles=[patch],fontsize=13)
ax.ax_joint.set_xlabel('log10(P)')
ax.ax_joint.set_ylabel('log10 of arbitrary feature importance')

#%%
y_pred = rfe_classifier.predict(X_test_selected)
y_pred_prob = rfe_classifier.predict_proba(X_test_selected)
misclassified = np.where(y_test != y_pred)[0]
classified = np.where(y_test == y_pred)[0]
#%%
f, p = test_features.test_features_sensitivity(data_dict, masses_list_raw, O_selected, param_grid, 'linear', 8)

#%%
fig, axes = subplots(4,6)
ax = axes.ravel()
for i in range(24):
    ax[i].plot(masses_list_raw, X_test_raw[misclassified[i]])
    ax[i].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[misclassified[i]],y_pred[misclassified[i]],y_pred_prob[misclassified[i]][0],
      y_pred_prob[misclassified[i]][1]))
    
fig, axes = subplots(4,6)
ax = axes.ravel()
for i in range(24):
    ax[i].plot(masses_list_raw, X_test_raw[misclassified[24+i]])
    ax[i].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[misclassified[24+i]],y_pred[misclassified[24+i]],y_pred_prob[misclassified[24+i]][0],
      y_pred_prob[misclassified[24+i]][1]))
    
fig, axes = subplots(4,6)
ax = axes.ravel()
for i in range(24):
    ax[i].stem(masses_list_raw[F_selected], X_test_selected[misclassified[i]],linefmt='k-',markerfmt='b.')
    ax[i].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[misclassified[i]],y_pred[misclassified[i]],y_pred_prob[misclassified[i]][0],
      y_pred_prob[misclassified[i]][1]))

fig, axes = subplots(4,6)
ax = axes.ravel()
for i in range(24):
    ax[i].stem(masses_list_raw[F_selected], X_test_selected[misclassified[i+24]],linefmt='k-',markerfmt='b.')
    ax[i].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[misclassified[i+24]],y_pred[misclassified[i+24]],y_pred_prob[misclassified[i+24]][0],
      y_pred_prob[misclassified[i+24]][1]))


#%%
fig, axes = subplots(4,6)
fig2, axes2 = subplots(4,6)
ax = axes.ravel()
ax2 = axes2.ravel()
index1 = 0
index2 = 0
for i in range(48):

    if y_test_raw[classified[i]] == 0:
        ax[index1].stem(masses_list_raw[F_selected], X_test_selected[classified[i]],linefmt='k-',markerfmt='b.')
        ax[index1].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[classified[i]],y_pred[classified[i]],y_pred_prob[classified[i]][0],
          y_pred_prob[classified[i]][1]))
        index1 = index1+1
    if y_test_raw[classified[i]] ==1:
        ax2[index2].stem(masses_list_raw[F_selected], X_test_selected[classified[i]],linefmt='k-',markerfmt='b.')
        ax2[index2].set_title('true:{}, pred:{}, prob_0:{:0.2f}, prob_1:{:0.2f}'.format(y_test_raw[classified[i]],y_pred[classified[i]],y_pred_prob[classified[i]][0],
          y_pred_prob[classified[i]][1]))
        index2 = index2+1

#%%
true_pos = X_test_selected[(y_test_raw == y_pred) & (y_test_raw==0)]
true_neg = X_test_selected[(y_test_raw == y_pred) & (y_test_raw==1)]

fig, axes = subplots(2,1)

ax = axes.ravel()

ax[0].stem(masses_list_raw[F_selected],true_pos.mean(0),linefmt='k-',markerfmt='b.')
ax[1].stem(masses_list_raw[F_selected],true_neg.mean(0),linefmt='k-',markerfmt='b.')
#%%
data_selected = intens_df_pp.iloc[:,O_selected]
corr = data_selected.corr()
from matplotlib import cm as cm
fig = plt.figure()
ax1 = fig.add_subplot(111)
cmap = cm.get_cmap('jet', 30)
cax = ax1.imshow(corr, interpolation="nearest", cmap=cmap)
labels = corr.columns
ax1.set_xticks(np.arange(0, len(labels)))
ax1.set_xticklabels(labels,fontsize=11,rotation='vertical')
ax1.set_yticklabels(corr.columns,fontsize=11)
ax1.set_yticks(np.arange(0, len(labels)))
fig.colorbar(cax)

#%%
fig, axes = subplots(1,2)

ax = axes.ravel()
ax[0].hist(y_pred_prob[:,0][y_test_raw==0],bins=20,color='c')
ax[0].set_title('astrocytes')
ax[0].set_ylabel('number of cells')
ax[0].set_xlabel('probability')
ax[1].hist(y_pred_prob[:,1][y_test_raw==1],bins=20,color='orange')
ax[1].set_title('neurons')
ax[1].set_ylabel('number of cells')
ax[1].set_xlabel('probability')