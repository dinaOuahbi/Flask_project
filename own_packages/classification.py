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
from sklearn.model_selection import train_test_split, LeaveOneOut

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier, VotingClassifier,GradientBoostingClassifier, StackingClassifier
from sklearn.neighbors import KNeighborsClassifier
#from xgboost.sklearn import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
#from lightgbm.sklearn import LGBMClassifier

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

#########################################################################
df = pd.read_csv('gex_preprocess_results/main_data_sc.csv', index_col = 0)
df.set_index('Patient', inplace=True)

########################### MODELS #####################################
#print(evaluation_skf.__doc__)



seed = 0
###########basic clf#############
cart_clf = DecisionTreeClassifier(random_state=seed)
knn_clf = KNeighborsClassifier(n_neighbors = 5,  n_jobs = 1)
#xgb_clf = XGBClassifier(random_state=seed)
svcl_clf = SVC(random_state=seed, probability=True, kernel='linear') #decision function
lr_clf = LogisticRegression(random_state =seed, dual = False, class_weight = None,  n_jobs = 1)

########BAGGING############
#le foule permet de reduir la variance car chaque model est en over fitting
rf_clf = RandomForestClassifier(n_jobs = 1, oob_score = True, n_estimators = 1000, verbose = 0, random_state =seed) 
estimators=[('lr', lr_clf), ('rf', rf_clf), ('cart', cart_clf)]
vot_clf = VotingClassifier(estimators=estimators,voting='soft')

#########BOOSTING###########
#chaque model est faible, en underfiting, la fool permet de reduir le biais
ab_clf = AdaBoostClassifier(random_state=seed)
gb_clf = GradientBoostingClassifier(n_estimators=100, learning_rate=1.0, max_depth=1, random_state=seed)

##########STACKING##############
#stacking clf
#stack_clf = StackingClassifier(estimators=estimators, final_estimator=xgb_clf)

#########LGBM################### 
#lgbm_clf = LGBMClassifier(n_jobs = 1, n_estimators = 1000, subsample = 0.8, colsample_bytree = 0.8, random_state =seed)

################################## LIST OF MACHINE LEARNING ALGORITHMS #######################
MLA = [cart_clf,
       knn_clf,
       svcl_clf,
       lr_clf,
       rf_clf,
       vot_clf,
       ab_clf,
       gb_clf]
 
from functions import evaluation_skf, preprocess, roc_curve_skf, learnCurve, mcc_score,  learnCurve, auc_roc_curve
############### split int target and features

X, y = preprocess(df)
skf = StratifiedKFold(n_splits=10)

############### evaluate model (default params, 5 seeds)     
'''default_models = pd.DataFrame(columns=['MCC_SCORE', 'AUC_SCORE', 'ACC_SCORE'])     
repitition = np.random.randint(0, 99999999, size = 5)      

for model in MLA:
    print(model.__class__.__name__)
    for seed in repitition:
       print('seed number : ',seed)
       default_models.loc['{} | {} '.format(model.__class__.__name__,seed)] = evaluation_skf(model, X, y, skf)
       print('_'*50)
       default_models.to_csv('classification_results/default_models.csv') '''    
       
       
################ ROC curves ###########################
'''plt.figure(figsize=(15,8))
for model in MLA:
    print(model.__class__.__name__)
    roc_curve_skf(model, X, y, skf,'classification_results')

plt.savefig('classification_results/default_model_skf_roc.png')'''

###################learning curve / interesting models ################

'''print(learnCurve.__doc__)
# op > input : model, X, y, cv / output : learning curve

learnCurve(cart_clf, X, y, skf,'classification_results')'''       
 
#####################
'''
loo = LeaveOneOut()
print('SPLITS NUMBER EQUAL TO ===> ',loo.get_n_splits(X))
default_models_loo = pd.DataFrame(index=['MCC_SCORE', 'ACC_SCORE'])
for model in MLA:
    print(model.__class__.__name__)
    l = []
    l = evaluation_skf(model, X, y, loo)
    default_models_loo[model.__class__.__name__] = l
    default_models_loo.to_csv('classification_results/default_models_loo.csv')'''

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
#print(X_ind_sc)     
'''          er_ihc  her2  drfs_even_time_years     NRIP1  ...      LY96      FZD5     SEMG2    RNF122
Patient                                                  ...                                        
GSM808102       1     0              2.057534  0.598751  ...  0.311100  0.567786  0.395897  0.478680
GSM808103       1     0              6.643836  0.482897  ...  0.347961  0.422442  0.336439  0.450877
'''
#print(X_ind)  
'''
          er_ihc  her2  drfs_even_time_years     NRIP1  ...      LY96      FZD5     SEMG2    RNF122
Patient                                                  ...                                        
GSM808102       1     0              2.057534  0.638788  ... -0.096269  0.235624  0.278190 -0.211947
GSM808103       1     0              6.643836  0.049577  ...  0.135953 -0.936995 -0.249146 -0.419212
'''     
#PREDICT Y_IND OF X_ind and X_ind_sc
'''
mcc_l = []
auc_l = []

mcc_sc = []
auc_sc = []

list_of_index = []

for model in MLA:
    print(model.__class__.__name__)
    list_of_index.append(model.__class__.__name__)
    ind_data_scores = pd.DataFrame(index=list_of_index)
    model.fit(X, y)
    ### predict original data
    y_pred = model.predict(X_ind)
    y_proba = model.predict_proba(X_ind)[:, 1]
    mcc_l.append(matthews_corrcoef(y_ind, y_pred))
    auc_l.append(roc_auc_score(y_ind, y_proba))
    ### predict scaled data
    y_pred_sc = model.predict(X_ind_sc)  
    y_proba_sc = model.predict_proba(X_ind_sc)[:, 1]
    mcc_sc.append(matthews_corrcoef(y_ind, y_pred_sc))
    auc_sc.append(roc_auc_score(y_ind, y_proba_sc))

ind_data_scores['mcc'] = mcc_l
ind_data_scores['auc'] = auc_l
ind_data_scores['mcc_sc'] = mcc_sc
ind_data_scores['auc_sc'] = auc_sc

ind_data_scores.to_csv('classification_results/ind_data_scores.csv')'''
    
# ROC CURES FOR INDEPENDENTS DATA
#input : model, X_train, y_train, X_test, y_test 
plt.figure(figsize=(15,8))
for model in MLA:
    print(model.__class__.__name__)
    auc_roc_curve(model, X, y, X_ind, y_ind)
plt.savefig('classification_results/ROC_curves_ind_data.png')
      
       
    
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
       
