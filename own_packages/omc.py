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
####################### dYSCOVERY DATA #####################################################
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
############################################################################################
'''mcc_score_fs = pd.DataFrame(columns=['MCC_SCORE', 'AUC_SCORE', 'Feature_number (k)'])
mcc_list = []
for alg in MLA:
    max_mcc = 0
    max_k = 0
    print('###################################',alg.__class__.__name__,'#####################################')
    for i in range(2,df.shape[0]//2):
        print('Number of features used : ',i)
        model = make_pipeline(SelectKBest(chi2, k=i), alg)
        mcc_k, auc = evaluation_skf(model, X, y, skf) #EVALUE chaque fold de cv avec un i donnee et retourne son mcc 
        mcc_list.append(mcc_k) # add mcc fold to list of mcc
        if mcc_k > max_mcc:
            max_mcc = mcc_k
            max_k = model[0] #select first pipeline step
        
        print('.'*10)
    mcc_score_fs.loc[alg.__class__.__name__] = max_mcc, auc, max_k
    mcc_score_fs.to_csv('OMC_results/mcc_auc_feature_selection.csv')'''
    
################################# plot omc ##################################################
#fluct_mcc = pd.DataFrame(index=range(2,df.shape[0]//2), columns=['mcc'])
'''
dict_of_mcc = {}
for alg in MLA[:5]:
    list_of_mcc = []
    for i in range (2,df.shape[0]//2):
        print('K = ',i, 'Model = ',alg.__class__.__name__)
        model = make_pipeline(SelectKBest(chi2, k=i),alg)
        mcc, acc = evaluation_skf(model, X, y, skf)
        list_of_mcc.append(mcc)
        print('accuracy = ',acc)
    
    dict_of_mcc[alg.__class__.__name__] = list_of_mcc
    print('best_mcc  => ',max(list_of_mcc))
    mcc_fluctuation_based_on_k = pd.DataFrame(dict_of_mcc, index = range(2,df.shape[0]//2))
    mcc_fluctuation_based_on_k.to_csv('OMC_results/mcc_fluctuation_based_on_k.csv')
    
plt.figure(figsize=(15,8))
for k, v in dict_of_mcc.items():
    plt.plot(dict_of_mcc[k], label=k, lw=2)
    plt.grid(True)
    plt.xlabel('number of features (k)')
    plt.ylabel('mcc-score')
    plt.legend()

plt.savefig('OMC_results/mcc_fluctuation_based_on_k.png')

'''

##################################### extract best feature ################################
'''
#extract best feature from df for each model
#mcc, auc = evaluation_skf(ab_fs, X, y)
list_of_k = [57,16,135,119,4,19,81,17,91,106,135]
dict_of_best_f = {}
for alg, k in zip(MLA, list_of_k): 
    print(alg.__class__.__name__, k)
    model = make_pipeline(SelectKBest(chi2, k=k),alg)
    mcc, auc = evaluation_skf(model, X, y, skf)
    temp_list = []
    for index in model.named_steps['selectkbest'].get_support(indices=True):
        temp_list.append(X.columns[index])
    dict_of_best_f[model[1].__class__.__name__] = temp_list

pd.DataFrame.from_dict(dict_of_best_f, orient='index').T.to_csv('OMC_results/index_best_feature.csv')
'''

################################### 
index_best_feature = pd.read_csv('OMC_results/index_best_feature.csv')
#print(index_best_feature.columns)

'''['DecisionTreeClassifier', 'KNeighborsClassifier', 'SVC','LogisticRegression', 'RandomForestClassifier', 'VotingClassifier','AdaBoostClassifier', 'GradientBoostingClassifier','StackingClassifier', 'LGBMClassifier', 'XGBClassifier']'''

'''
list_of_index = []
mcc = []
mcc_sc = []
auc = []
auc_sc = []

for alg in MLA:
    print(alg.__class__.__name__)
    list_of_index.append(alg.__class__.__name__)
    omc_ind = pd.DataFrame(index = list_of_index)
    
    #extract variables from independing data
    X_ind_temp = X_ind[index_best_feature[alg.__class__.__name__].dropna()]
    X_ind_sc_temp = X_ind_sc[index_best_feature[alg.__class__.__name__].dropna()]
    
    print('shape : ',X_ind_temp.shape)
    
    #train model on entire discovery data (X, y)
    
    alg.fit(X[index_best_feature[alg.__class__.__name__].dropna()], y)
    
    #predict independing data (X_ind and X_ind scaled)
    y_pred = alg.predict(X_ind_temp)
    y_pred_sc = alg.predict(X_ind_sc_temp)
    #predict proba 
    y_proba = alg.predict_proba(X_ind_temp)[:, 1]
    y_proba_sc = alg.predict_proba(X_ind_sc_temp)[:, 1]

    #metrics (mcc and auc)
    mcc.append(matthews_corrcoef(y_ind, y_pred))
    mcc_sc.append(matthews_corrcoef(y_ind, y_pred_sc))
    auc.append(roc_auc_score(y_ind, y_proba))
    auc_sc.append(roc_auc_score(y_ind, y_proba_sc))
#je stock mes score dans la dataframe
omc_ind['mcc'] = mcc
omc_ind['mcc_sc'] = mcc_sc
omc_ind['auc'] = auc
omc_ind['auc_sc'] = auc_sc
    
omc_ind.to_csv('OMC_results/omc_ind.csv')

'''

############################################## compart with all feature #####################
'''
ccl_tab = pd.read_csv('OMC_results/ccl_table_ind.csv')
print(ccl_tab)

barWidth = 0.4
y1 = ccl_tab['mcc_sc(omc)']
y2 = ccl_tab['mcc_sc(af)']
r1 = range(len(y1))
r2 = [x + barWidth for x in r1]

plt.bar(r1, y1, width = barWidth, color = ['yellow' for i in y1],
           edgecolor = ['blue' for i in y1], linewidth = 1, label='OMC')
plt.bar(r2, y2, width = barWidth, color = ['blue' for i in y1],
           edgecolor = ['red' for i in y1], linewidth = 1, label='All-features')
plt.xticks([r + barWidth / 2 for r in range(len(y1))], ccl_tab['models'].tolist(), rotation=90)
plt.legend()
plt.ylabel('Mcc_score')
plt.title('MCC obtained from independent data scaled \n Default threshold')
plt.grid(True)
#plt.show()
plt.savefig('OMC_results/compart_omc_af_ind_sc.png')

'''































