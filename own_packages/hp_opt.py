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
       
####################### DATA ##############################
df = pd.read_csv('gex_preprocess_results/main_data_sc.csv', index_col = 0)
df.set_index('Patient', inplace=True)
X, y = preprocess(df)
skf = StratifiedKFold(n_splits=10) 

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
############################################## define params #######################################


knn_hp = {'n_neighbors': list(range(1,31))}

ab_hp = {'n_estimators':[10,50,250,1000],'learning_rate':[0.01,0.1]}
         
gb_hp = {'n_estimators':[1, 2, 4, 8, 16, 32, 64, 100, 200],'max_depth':np.linspace(1, 32, 32, endpoint=True),'min_samples_split':np.linspace(0.1, 1.0, 10, endpoint=True),'min_samples_leaf': np.linspace(0.1, 0.5, 5),'max_features':["log2","sqrt"]}

stack_hp = {'final_estimator':[xgb_clf, gb_clf, ab_clf]}

lgbm_hp = {'learning_rate': [0.005, 0.01],'n_estimators': [8,16,24],'max_depth':range(2,20),'min_child_weight':range(1,5),'num_leaves':range(20,200),'class_weilght':['balanced',None]}
    
xgb_hp = {'min_child_weight': [1, 5, 10],'gamma': [0.5, 1, 1.5, 2, 5],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'max_depth': [3, 4, 5]}

model_to_opt = [knn_clf, ab_clf, gb_clf,stack_clf, lgbm_clf, xgb_clf]

dict_of_hp = {'KNeighborsClassifier':knn_hp, 'AdaBoostClassifier':ab_hp, 'GradientBoostingClassifier':gb_hp, 'StackingClassifier':stack_hp, 'LGBMClassifier':lgbm_hp, 'XGBClassifier':xgb_hp}
############################################ GRIDSEARCH CV ##########################################
index_best_feature = pd.read_csv('OMC_results/index_best_feature.csv')

list_of_index = []
mcc_scorer = make_scorer(matthews_corrcoef, greater_is_better=True)
hp_opt = pd.DataFrame(index = (model.__class__.__name__ for model in model_to_opt))
for model in model_to_opt:
    best_params = []
    print(model.__class__.__name__)
    X_temp = X[index_best_feature[model.__class__.__name__].dropna()]
    grid = GridSearchCV(estimator=model,
                         param_grid=dict_of_hp[model.__class__.__name__],
                         scoring=mcc_scorer,
                         verbose=10,
                         n_jobs=30,
                         cv = skf)
    grid.fit(X_temp, y)
    best_params.append(grid.best_estimator_.get_params())
    mcc = grid.best_score_ 
    hp_opt.loc[model.__class__.__name__,'best_params'] = best_params
    hp_opt.loc[model.__class__.__name__,'Mcc_scorer'] = mcc
    hp_opt.to_csv('parametres_opt_results/hp_opt.csv')











































































    
 
