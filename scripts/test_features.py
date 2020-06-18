# SVM
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn import svm
from sklearn.svm import SVC
from sklearn import linear_model
from sklearn.metrics import classification_report, confusion_matrix  
from sklearn.externals import joblib
from sklearn.metrics import roc_curve, auc
from scipy.stats import ranksums
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.pyplot import subplots,scatter
import seaborn as sns
matplotlib.interactive(True)


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


def SVM_classi(X_train, y_train, kernel_, param_grid, if_probability, cores):
    #     X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    #     data_dict = {'X_train':X_train,'X_test':X_test,'y_train':y_train,'y_test':y_test}

    grid_search = GridSearchCV(svm.SVC(kernel=kernel_), param_grid, cv=5, n_jobs=cores)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    svclassifier = SVC(kernel=kernel_, C=best_params['C'], gamma=best_params['gamma'],probability=if_probability)
    svclassifier.fit(X_train, y_train)

    return svclassifier

def LR_classi(X_train, y_train, param_grid, cores):
    
    grid_search = GridSearchCV(linear_model.LogisticRegression(), param_grid, cv=10, n_jobs=cores)

    grid_search.fit(X_train, y_train)
    best_params = grid_search.best_params_

    lrclassifier = linear_model.LogisticRegression(C=best_params['C'], penalty=best_params['penalty'])
    lrclassifier.fit(X_train, y_train)

    return lrclassifier

def predict(X_test, y_test, classifier):
    y_pred = classifier.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report_dict = classification_report(y_test, y_pred, output_dict=True)
    # sns.heatmap(cm,annot=True,fmt="d")
    return y_pred, cm, report_dict


def roc(classifier, X_test, y_test, n_classes):
    y_score = classifier.decision_function(X_test)

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    fpr['0'], tpr['0'], _ = roc_curve(y_test, y_score)
    roc_auc['0'] = auc(fpr['0'], tpr['0'])

    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

    return fpr, tpr, roc_auc

from sklearn.metrics import roc_curve, auc

def feature_sampling(data,label,model,feature_num_sampled,sampling_num):
    
    label_ = label.copy()
    label_ = np.asarray(label_,int)
    acc = []
    recall = []
    f1 = []
    auc_ = []
    
    for i in range(sampling_num):
        data_sampled = data.sample(feature_num_sampled,axis=1)
        X_train, X_test, y_train, y_test = train_test_split(data_sampled.values, label_, test_size=0.2,random_state=19)
        
        model.fit(X_train,y_train)

        y_pred = model.predict(X_test)
        y_pred_prob = model.predict_proba(X_test)
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        acc.append(report_dict['accuracy'])
        recall.append(report_dict['macro avg']['recall'])
        f1.append(report_dict['macro avg']['f1-score'])
        
        a,b,c=roc_curve(y_test,y_pred_prob[:,1])
        print('auc: {}'.format(auc(a,b)))
        auc_.append(auc(a,b))
        
    return acc,recall,f1,auc_

def feature_select(data,label,model,ranked_index,steps):
    
    label_ = label.copy()
    label_ = np.asarray(label_,int)
    acc = []
    recall = []
    f1 = []
    auc_ = []
    
    for i in range(1,100):
        X_train, X_test, y_train, y_test = train_test_split(data.values, label_, test_size=0.2,random_state=19)
        model.fit(X_train[:,ranked_index[:i]],y_train)

        y_pred = model.predict(X_test[:,ranked_index[:i]])
        y_pred_prob = model.predict_proba(X_test[:,ranked_index[:i]])
        report_dict = classification_report(y_test, y_pred, output_dict=True)
        acc.append(report_dict['accuracy'])
        recall.append(report_dict['macro avg']['recall'])
        f1.append(report_dict['macro avg']['f1-score'])
        
        a,b,c=roc_curve(y_test,y_pred_prob[:,1])
        print('auc: {}'.format(auc(a,b)))
        auc_.append(auc(a,b))
        
    return acc,recall,f1,auc_
