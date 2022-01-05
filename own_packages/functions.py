from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve,auc
import math
from scipy import stats
#HANDLING AND PLOTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#STATISTIC TEST
from scipy.stats import ttest_ind, mannwhitneyu

#SCALING 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures

#SPLITTING
from sklearn.model_selection import train_test_split

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, VotingClassifier,GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMClassifier

#METRICS
from sklearn.metrics import f1_score, confusion_matrix, classification_report,accuracy_score, matthews_corrcoef, roc_auc_score,auc

#WETHER OVERFITING
from sklearn.model_selection import learning_curve, StratifiedKFold, GridSearchCV

#FEATURE SELECTION
from sklearn.feature_selection import SelectKBest, f_classif, chi2

#pipeline
from sklearn.pipeline import make_pipeline

#warning avoid
import warnings
warnings.filterwarnings("ignore")

from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer, make_column_selector
import pickle
####################################### fonctions definies ####################################

####################################################################### PREPROCESS DATA #######################################
def preprocess(df):
    '''input : df / output : X, y'''
    X = df.drop('RECIST', axis=1)
    y = df['RECIST']   
    return X,y

def mcc_score(y_true, y_pred):
    '''input y_true, y_pred / output mcc score'''
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc
    
    
####################################################################### TIMER #######################################
def timer(start_time=None):
    
    '''cette fonction calcule temps ecouler d un programme en heurs, minute et seconde 
    start_time = timer(None)
    timer(start_time)'''
    
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        return '\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2))
        
####################################################################### EVALUATION SKF #######################################

def evaluation_skf(model, X, y, skf):
    '''input : model, X, y / output : mcc_avg, auc_avg, acc_avg#, model.best_estimator_'''
    y_proba_list = []
    print('CLASSIFIER :' ,model.__class__.__name__)
    print('NUMBER OF SPLIT : ',skf.get_n_splits(X,y))
    mcc = []
    acc = []
    auc = []
    for train_index, test_index in skf.split(X,y):
        #print("TRAIN:", train_index, "TEST:", test_index)
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        y_proba_list.append(y_proba)
        mcc.append(matthews_corrcoef(y_test, y_pred))
        acc.append(accuracy_score(y_test, y_pred))
        try:
            auc.append(roc_auc_score(y_test, y_proba))
        except ValueError:
            pass
    
    auc_avg = np.average(auc)
    mcc_avg = np.average(mcc)
    acc_avg = np.average(acc)
    y_proba_list = np.concatenate(y_proba_list)
    #print('MCC ======================> ',mcc_avg)
    #print('ACC ======================>',acc_avg)
    #print('AUC ======================>',auc_avg)
    #return y_proba
    return mcc_avg, acc_avg, auc_avg
      
####################################################################### EVALUATION X_TRAIN #######################################
def evaluation(model, X_train, X_test):
    '''input : model, X_train, X_test / output : cm, cr, mcc,auc'''
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    #print(pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'],margins=True))
    cr = classification_report(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)
    auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
    return cm, cr, mcc, auc
    
####################################################################### LEARNING CURVE CV ####################################
def learnCurve(model, X, y, cv, folder):
    '''input : model, X, y, cv / output : learning curve'''
    mcc_scorer=make_scorer(matthews_corrcoef, greater_is_better=True)
    train_sizes, train_scores, val_scores = learning_curve(model,X,y,cv=cv,scoring=mcc_scorer,shuffle=True,train_sizes=np.linspace(0.1,1.0,200),verbose = 0)
    #plot learning curve
    plt.figure(figsize=(10,5))
    plt.plot(train_sizes, val_scores.mean(axis=1), label='Validation score', lw=3)
    plt.plot(train_sizes, train_scores.mean(axis=1), label='Training score', lw=3)
    plt.xlabel('Partitions of data')
    plt.ylabel('mcc_score')
    plt.title(model.__class__.__name__)
    plt.legend()
    plt.grid(True)   
    plt.savefig('{}/{}_learning curve'.format(folder, model.__class__.__name__))
    
    

####################################################################### ROC CURVE TRAIN TEST #######################################
def auc_roc_curve(model, X_train, y_train, X_test, y_test):
    '''input : model, X_train, y_train, X_test, y_test
       OUTPUT : ROC curve '''
    probas_ = model.fit(X_train, y_train).predict_proba(X_test)
    fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
    auc = roc_auc_score(y_test, probas_[:, 1])
    
    plt.plot(fpr, tpr, label='{}_auc = {}'.format(model.__class__.__name__, auc.round(3)), lw=2)
    plt.plot([0, 1], [0, 1], color='blue', linestyle='--')
    plt.grid(True)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Roc_curves')
    plt.legend(loc=4)
    
    
####################################################################### ROC CURVE SKF #######################################
def roc_curve_skf(model, X, y, skf, folder):
    tprs = []
    aucs = []
    #best_th_list = []
    #best_gmeans_list = []
    mean_fpr = np.linspace(0, 1, 100)
    #plt.figure(figsize=(10,5))
    i = 0
    for train_index, test_index in skf.split(X,y):
        #pour chaque fold, le programe trace une courbe roc
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        probas_ = model.fit(X_train, y_train).predict_proba(X_test)
        fpr, tpr, thresholds = roc_curve(y_test, probas_[:, 1])
        
        #print('threshold fold : ',thresholds)
        #calcule the gmean (as list) for each threshold
        gmeans = np.sqrt(tpr * (1-fpr))
        ix = np.argmax(gmeans)
        #print('best_threshold = {}, G_means = {}'.format(thresholds[ix], gmeans[ix]))
        #best_th_list.append(thresholds[ix])
        #best_gmeans_list.append(gmeans[ix])

        tprs.append(np.interp(mean_fpr, fpr, tpr))
        tprs[-1][0] = 0.0
        roc_auc = auc(fpr, tpr)
        aucs.append(roc_auc)
        #plt.plot(fpr, tpr, lw=1, alpha=0.3,label='ROC fold %d (AUC = %0.2f)' % (i, roc_auc))
        #plt.scatter(fpr[ix], tpr[ix], marker='o', color='black')
        i += 1
    plt.plot([0, 1], [0, 1], linestyle='--', lw=2, color='r',alpha=.8)

    mean_tpr = np.mean(tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = auc(mean_fpr, mean_tpr)
    std_auc = np.std(aucs)
    #print('best_threshold_mean = {}, G_means_mean = {}'.format(np.mean(best_th_list), np.mean(best_gmeans_list)))
    plt.plot(mean_fpr, mean_tpr,label='{}, Mean AUC {}, std_auc {}'.format(model.__class__.__name__,round(mean_auc,2),round(std_auc,2)),lw=2, alpha=.8)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC_Curves')
    plt.legend(loc="lower right")
    #plt.text(0.32,0.7,'More accurate area',fontsize = 12)
    #plt.text(0.63,0.4,'Less accurate area',fontsize = 12)
    plt.grid(True)
    #plt.savefig('{}/{}_ROC_curve_SKF.png'.format(folder, model.__class__.__name__))
    
    
    
 ####################################################################### decision function #######################################   
def DecisionFunction(clf):
    '''input : clf / output : precision , recall , threshold, precision-recall curve'''
    Decision_Function = clf.decision_function(X_test)
    precision, recall, threshold = precision_recall_curve(y_test, Decision_Function)
    plt.figure()
    plt.plot(threshold, precision[:-1], label='Precision', lw=4, color='red')
    plt.plot(threshold, recall[:-1], label='Recall', lw=4, ls='--', color='blue')
    plt.xlabel('{} Threshold'.format(clf))
    plt.ylabel('Score')
    plt.title('Decision function | {}'.format(clf.__class__.__name__))
    plt.grid(True)
    plt.legend()
    plt.savefig('{} / DECISION_FUNCTION '.format(clf.__class__.__name__))
    return precision, recall, threshold
    
####################################################################### ROC CURVE SKF #######################################   
def paramsTuning(model_to_opt):

    list_of_be = []
    list_of_best_mcc = []
    optimize_model = pd.DataFrame(index=model_to_opt)
    for model in model_to_opt:
        print(model.__class__.__name__)
        X_temp = X_df[dict_of_best_f[model.__class__.__name__]]
        grid = GridSearchCV(estimator=model,param_grid=dict_hp[model.__class__.__name__],scoring=mcc_scorer,verbose=10,n_jobs=1,cv = skf)
        grid.fit(X_temp, y_df)
        list_of_be.append(grid.best_estimator_.get_params()) #LIST OF 9 PARAMETERS
        list_of_best_mcc.append(grid.best_score_) # I HAVE TO GOT A LIST OF 9 SCORES
    
    
    optimize_model['best_estimator'] = list_of_be
    optimize_model['best_mcc'] = list_of_best_mcc
    return optimize_model

####################################################################### ROC CURVE SKF #######################################      
def select_adjusted_threshold(y, y_proba):
    thresholds = [0., 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.]
    l_tuples_thr_mcc = []
    
    #calcule les mcc pour tous les probabilities et les ajoutent dans la table 
    for t in thresholds:
        y_pred_thr = np.where(y_proba[:,1] > t, 1, 0)
        mcc_thr = matthews_corrcoef(y, y_pred_thr)
        l_tuples_thr_mcc.append((t, mcc_thr, y_pred_thr))
        
    return l_tuples_thr_mcc
    
    '''try:
        tup_thr_with_highest_mcc = sorted([tup for tup in l_tuples_thr_mcc if tup[1] == np.nanmax([x[1] for x in l_tuples_thr_mcc])], key = lambda x:x[0])[0]
        
        except IndexError:
            for tup in l_tuples_thr_mcc:
                if tup[0] == 0.5:
                    tup_thr_with_highest_mcc = tup
                    
        thr_with_highest_mcc = tup_thr_with_highest_mcc[0]
        highest_mcc = tup_thr_with_highest_mcc[1]
        pre_thr_highest_mcc = precision_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
        rec_thr_highest_mcc = recall_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
        f1_thr_highest_mcc = f1_score(y, tup_thr_with_highest_mcc[2], average = 'weighted')
        accu_thr_highest_mcc = accuracy_score(y, tup_thr_with_highest_mcc[2])
        
        return thr_with_highest_mcc, highest_mcc, pre_thr_highest_mcc, rec_thr_highest_mcc, f1_thr_highest_mcc, accu_thr_highest_mcc''' 
        
        
# apply threshold to positive probabilities to create labels
def to_labels(pos_probs, threshold):
	return (pos_probs >= threshold).astype('int')       
         
