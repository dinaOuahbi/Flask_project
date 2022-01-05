from sklearn.metrics import make_scorer
from sklearn.metrics import roc_curve,auc
import math
from scipy import stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind, mannwhitneyu
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, VotingClassifier,GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from lightgbm.sklearn import LGBMClassifier
from sklearn.metrics import f1_score, confusion_matrix, classification_report,accuracy_score, matthews_corrcoef, roc_auc_score,auc
from sklearn.model_selection import learning_curve, StratifiedKFold, GridSearchCV
from sklearn.feature_selection import SelectKBest, f_classif, chi2
from sklearn.pipeline import make_pipeline
import warnings
warnings.filterwarnings("ignore")
from datetime import datetime
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
from sklearn.decomposition import PCA
from sklearn.compose import make_column_transformer, make_column_selector
import pickle
from functions import evaluation_skf, preprocess, roc_curve_skf, learnCurve, mcc_score,  learnCurve, auc_roc_curve
from numpy import sqrt
from numpy import argmax
########################### MODELS ##################################################################################
#print(evaluation_skf.__doc__)
seed = 0
cart_clf = DecisionTreeClassifier(random_state=seed)
knn_clf = KNeighborsClassifier(n_neighbors = 5,  n_jobs = 1)
xgb_clf = XGBClassifier(random_state=seed)
svcl_clf = SVC(random_state=seed, probability=True, kernel='linear') #decision function
lr_clf = LogisticRegression(random_state =seed, dual = False, class_weight = None,  n_jobs = 1)
rf_clf = RandomForestClassifier(n_jobs = 1, oob_score = True, n_estimators = 1000, verbose = 0, random_state =seed) 
estimators=[('lr', lr_clf), ('rf', rf_clf), ('cart', cart_clf)]
vot_clf = VotingClassifier(estimators=estimators,voting='soft')
ab_clf = AdaBoostClassifier(random_state=seed)
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=seed)
stack_clf = StackingClassifier(estimators=estimators, final_estimator=xgb_clf)
lgbm_clf = LGBMClassifier(n_jobs = 1, n_estimators = 1000, subsample = 0.8, colsample_bytree = 0.8, random_state =seed)
MLA = [cart_clf,
       knn_clf,
       svcl_clf,
       lr_clf,
       rf_clf,
       vot_clf,
       ab_clf,
       gb_clf,
       stack_clf,
       lgbm_clf,
       xgb_clf]
####################### MAIN DATA ###################################################
df = pd.read_csv('gex_preprocess_results/main_data_sc.csv', index_col = 0)
df.set_index('Patient', inplace=True)
X, y = preprocess(df)
skf = StratifiedKFold(n_splits=10) 
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.25, random_state=2, stratify=y)
################################### INDEPENDENT DATA ####################################

#loading 
ind = pd.read_csv('datas/C07_clinical.csv')    
#set patient id as index
ind.set_index('Patient', inplace=True)

#split data into features and target
X_ind, y_ind = preprocess(ind)
#print('X_ind_shape : ',X_ind.shape,'y_ind shape : ',y_ind.shape) 

#minmax scaler on genetic expression data : 
ind_gex = X_ind[X_ind.columns[3:]]
ind_clin = X_ind[X_ind.columns[:3]]
#TRANSFORMERS genes expression into scaled data
scaler = MinMaxScaler()
ind_gex_sc = pd.DataFrame(scaler.fit_transform(ind_gex.values), columns=ind_gex.columns, index=ind_gex.index)

#get the independing data scaling just on expression data
X_ind_sc = pd.merge(ind_clin, ind_gex_sc, on = ind_clin.index).rename(columns={'key_0':'Patient'}).set_index('Patient')   

'''
Now i have two types oF X_ind : 
	===> X_ind 
	===> X_ind_sc 

'''
############################Optimal Threshold for ROC Curve###############################

#The curve is useful to understand the trade-off in the true-positive rate and false-positive rate for different thresholds.
#AUC = summarize the performance of a model (0.5 = NO-SKILL / 1 = PERFECT SKILL)

#BEST TH = Il s'agirait d'un seuil sur la courbe qui est la plus proche du haut et de la gauche du graphique.

#select the threshold with the largest G-Mean value.
#===> G-Mean = sqrt(Sensitivity * Specificity)  #recherche un meilleur equilibre entre sens et spec

#split into train/ test sets


'''
list_of_index = []
th_l = []
gmean_l = []
plt.figure(figsize=(15,8))
temp = 0
for model in MLA:
    print(model.__class__.__name__)
    list_of_index.append(model.__class__.__name__)
    opt_t_df = pd.DataFrame(index = list_of_index)
    #fit a model
    model.fit(trainX, trainy)
    #predict probabilities
    probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only (keep only the probability predictions for the minority class)
    probs = probs[:, 1]
    #calculate AUC score
    auc = roc_auc_score(testy, probs)
    #calculate roc curves
    fpr, tpr, thresholds = roc_curve(testy, probs)
    # calculate the g-mean for each threshold
    gmeans = sqrt(tpr * (1-fpr))
    # locate the index of the largest g-mean
    ix = argmax(gmeans)
    print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    th_l.append(thresholds[ix])
    gmean_l.append(gmeans[ix])
    # plot the roc curve for the model
    while(temp==0):
        plt.plot([0,1], [0,1], linestyle='--', label='No Skill')
        temp+=1
    plt.plot(fpr, tpr, label='{}|auc={}'.format(model.__class__.__name__,round(auc, 2)))
    plt.scatter(fpr[ix], tpr[ix], marker='o', color='black')
    # axis labels
    plt.xlabel('False Positive Rate(Specificity)')
    plt.ylabel('True Positive Rate(Sensitivity)')
    plt.legend()

plt.savefig('threshold_optimization_results/dm_optT_roc.png')
opt_t_df['opt_thresholds'] = th_l
opt_t_df['ROC_gmean'] = gmean_l
opt_t_df.to_csv('threshold_optimization_results/opt_t_df.csv') 

'''
##########################Optimal Threshold for Precision-Recall Curve#####################


#a precision-recall curve focuses on the performance of a classifier on the positive (minority class) only.
#A no-skill model is represented by a horizontal line with a precision that is the ratio of positive examples in the dataset (e.g. TP / (TP + TN)), or 0.01 on our synthetic dataset. perfect skill #classifier has full precision and recall with a dot in the top-right corner



plt.figure(figsize=(10,15))
fscore_l = []
th_l = []
temp = 0
list_of_index = []
for model in MLA:
    print(model.__class__.__name__)
    list_of_index.append(model.__class__.__name__)
    optT_prec_rec = pd.DataFrame(index = list_of_index)
    model.fit(trainX, trainy)
    # predict probabilities
    probs = model.predict_proba(testX)
    # keep probabilities for the positive outcome only
    probs = probs[:, 1]
    # calculate pr-curve
    precision, recall, thresholds = precision_recall_curve(testy, probs)
    # convert to f score
    fscore = (2 * precision * recall) / (precision + recall)
    # locate the index of the largest f score
    ix = argmax(fscore)
    print('Best Threshold=%f, F-Score=%.3f' % (thresholds[ix], fscore[ix]))
    fscore_l.append(fscore[ix])
    th_l.append(thresholds[ix])
    # plot the roc curve for the model
    
    while(temp==0):
        no_skill = len(testy[testy==1]) / len(testy)
        plt.plot([0,1], [no_skill,no_skill], linestyle='--', label='No Skill')
        temp+=1
    plt.plot(recall, precision, marker='.', label='{} | f1-score = {}'.format(model.__class__.__name__,round(fscore[ix],2)))
    plt.scatter(recall[ix], precision[ix], marker='o', color='black')
    # axis labels
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-recall curves of testing-set\n the point denote the maximal F1-score')
    plt.grid(True)
    plt.legend()
  
#plt.show()
plt.savefig('threshold_optimization_results/optT_prec_rec.png')
optT_prec_rec['opt_thresholds'] = th_l
optT_prec_rec['F-score'] = fscore_l
optT_prec_rec.to_csv('threshold_optimization_results/optT_prec_rec.csv')



##########################Optimal Threshold Tuning#####################
'''
# define thresholds
thresholds = np.arange(0, 1, 0.001)
# apply threshold to positive probabilities to create labels
from functions import to_labels

opt_t = []
corr_f1 = []
opt_mcc = []
list_of_index = []
# fit a model
for model in MLA:
    f1_scores = []
    mcc_scores = []
    list_of_index.append(model.__class__.__name__)
    optT_tuning = pd.DataFrame(index = list_of_index)
    print(model.__class__.__name__)
    model.fit(trainX, trainy)
# predict labels
    yhat = model.predict(testX)
# evaluate the model
    score = f1_score(testy, yhat)
    print('F-Score: %.5f' % score)

# predict probabilities
    probs = model.predict_proba(testX)[:,1]
# evaluate each threshold

    for t in thresholds:
        f1_scores.append(f1_score(testy, to_labels(probs, t)))
        mcc_scores.append(matthews_corrcoef(testy, to_labels(probs, t)))
# get best threshold
    ix = argmax(mcc_scores)
    opt_t.append(thresholds[ix])
    corr_f1.append(f1_scores[ix])
    opt_mcc.append(mcc_scores[ix])
    #print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))

optT_tuning['opt_th'] = opt_t
optT_tuning['opt_mcc'] = opt_mcc
optT_tuning['corr_f1'] = corr_f1

optT_tuning.to_csv('threshold_optimization_results/optT_tuning.csv')

'''
######################## ccl plot _ compart default T / opt T / opt K #####################
'''
ccl_tab = pd.read_csv('threshold_optimization_results/ccl_table.csv')
print(ccl_tab)

barWidth = 0.4
y1 = ccl_tab['default _t_mcc']
y2 = ccl_tab['opt_t_mcc']
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, y1, width = barWidth, color = ['yellow' for i in y1],
           edgecolor = ['blue' for i in y1], linewidth = 1, label='All-features\nDefault threshold')
plt.bar(r2, y2, width = barWidth, color = ['blue' for i in y1],
           edgecolor = ['red' for i in y1], linewidth = 1, label='All-features\nOptimal threshold')
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ccl_tab['models'].tolist(), rotation=90)
plt.legend()
plt.ylabel('Mcc_score')
plt.title('MCC obtained from Testing data')
plt.grid(True)
#plt.show()
plt.savefig('threshold_optimization_results/compart_defT_optT.png')


'''
######################################### opt threshold on independing data ###############
'''
opt_th = pd.read_csv('threshold_optimization_results/optT_tuning.csv')
print(opt_th)
list_of_index = []
ind_mcc = []
ind_f1 = []
ind_sc_mcc = []
ind_sc_f1 = []

for model, t in zip(MLA, opt_th['opt_th']):
    print(model.__class__.__name__, t)
    list_of_index.append(model.__class__.__name__)
    opt_t_ind_data = pd.DataFrame(index = list_of_index)
    model.fit(X, y)
# predict labels
    y_pred = model.predict(X_ind)
    y_pred_sc = model.predict(X_ind_sc)
# evaluate the model
    f1 = f1_score(y_ind, y_pred)
    f1_sc = f1_score(y_ind, y_pred_sc)
    mcc = matthews_corrcoef(y_ind, y_pred)
    mcc_sc = matthews_corrcoef(y_ind, y_pred_sc)
    print('F-Score_sc:{} | F-score:{}'.format(f1_sc,f1))
    print('mcc-Score_sc:{} | mcc-score:{}'.format(mcc_sc,mcc))

# predict probabilities
    probs = model.predict_proba(X_ind)[:,1]
    probs_sc = model.predict_proba(X_ind_sc)[:,1]
# evaluate opt threshold
    ind_mcc.append(matthews_corrcoef(y_ind, to_labels(probs, t)))
    ind_f1.append(f1_score(y_ind, to_labels(probs, t)))
    ind_sc_mcc.append(matthews_corrcoef(y_ind, to_labels(probs_sc, t)))
    ind_sc_f1.append(f1_score(y_ind, to_labels(probs_sc, t)))

opt_t_ind_data['opt_th'] = opt_th['opt_th']
opt_t_ind_data['independent_mcc'] = ind_mcc
opt_t_ind_data['independent_f1'] = ind_f1
opt_t_ind_data['independent_sc_mcc'] = ind_sc_mcc
opt_t_ind_data['independent_sc_f1'] = ind_sc_f1


opt_t_ind_data.to_csv('threshold_optimization_results/opt_t_ind_data.csv')

####################################### independent data #################################

ccl_tab_ind = pd.read_csv('threshold_optimization_results/ccl_table_ind.csv')
print(ccl_tab_ind)

barWidth = 0.4
y1 = ccl_tab_ind['ind_sc(defaultT)']
y2 = ccl_tab_ind['ind_sc(optT)']
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, y1, width = barWidth, color = ['yellow' for i in y1],
           edgecolor = ['blue' for i in y1], linewidth = 1, label='Default_threshold')
plt.bar(r2, y2, width = barWidth, color = ['green' for i in y1],
           edgecolor = ['blue' for i in y1], linewidth = 1, label='optimal T')
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ccl_tab_ind['model'].tolist(), rotation=90)
plt.legend()
plt.xlabel('Machine Learning ALgorithm')
plt.ylabel('Mcc_score')
plt.title('MCC obtained from all-features models \n Independent data scaled')
plt.grid(True)
#plt.show()
plt.savefig('threshold_optimization_results/compart_defT_ind_sc.png')

'''




