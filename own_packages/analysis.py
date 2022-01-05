
###############################################CLINICAL DATA ANALYSIS ##########################
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#STATISTIC TEST
from scipy.stats import ttest_ind, mannwhitneyu

#warning avoid
import warnings
warnings.filterwarnings("ignore")



###################################################### LOAD CLINICAL DATA #######################
cd = pd.read_csv('datas/05_hatzis_2011_clinical.csv')

#we'll create a copy
cd1 = cd.copy()

###################################################### DATA HUNDLING ############################
cd.set_index('patient_ID', inplace=True)
cd.drop('NAC', axis=1, inplace=True)
cd = cd[['age', 'er_ihc', 'pr_ihc', 'her2', 'drfs_even_time_years', 'RECIST']]

#ENCODING
dico = {' P':0, ' N':1, ' I':np.nan, ' NA':np.nan}
cd['er_ihc'] = cd['er_ihc'].map(dico)
cd['pr_ihc']=cd['pr_ihc'].map(dico)
cd['her2']=cd['her2'].map(dico)
cd['RECIST'] = cd['RECIST'].map({' RD':0, ' pCR':1, ' NA':np.nan}) 

#TYPES CONVERSION
cd['er_ihc']=cd['er_ihc'].astype('object')
cd['pr_ihc']=cd['pr_ihc'].astype('object')
cd['her2']=cd['her2'].astype('object')
cd['RECIST']=cd['RECIST'].astype('object')

#replace by most common values
cd['er_ihc'].fillna(cd['er_ihc'].value_counts().index[0], inplace=True)
cd['pr_ihc'].fillna(cd['pr_ihc'].value_counts().index[0], inplace=True)
cd['her2'].fillna(cd['her2'].value_counts().index[0], inplace=True)

#remove patient who don't have response (4 patients)
cd.dropna(inplace=True)

NR_df = cd[cd['RECIST'] == 0]
R_df = cd[cd['RECIST'] == 1]

#inbalance data
balanced_NR = NR_df.sample(n= R_df.shape[0], random_state=0)

######################################################### PRINT #################################
print('cd_shape = \n',cd.shape)
print('target_count = \n',cd['RECIST'].value_counts())
print('er_ihc_unique = \n',cd['er_ihc'].unique())
print('pr_ihc_unique = \n',cd['pr_ihc'].unique())
print('her2_ihc_unique = \n',cd['her2'].unique())
print('recist_ihc_unique = \n',cd['RECIST'].unique())
print('Types of variables : \n',cd.dtypes)
print('missing values = \n',(cd.isna().sum()).sort_values(ascending=False))




######################################################## PLOTING ###############################
#nan en heatmap / affichage de tous les patient avec yticklabels
sns.heatmap(cd.isna(), yticklabels=True)

#distribution variables categorielles
for objet in cd.select_dtypes('object'):
    plt.figure()
    cd[objet].value_counts().plot.pie(autopct = lambda x: str(round(x, 2)) + '%', shadow=True)
    plt.savefig('cd_analysis_results/{}_dist.png'.format(object))
 
#dist dsf
plt.hist('drfs_even_time_years', bins=100, data=cd, color='green')
plt.grid(True)
plt.xlabel('PFS(years)')
plt.ylabel('Count')
plt.title('PFS destribution')
plt.savefig('cd_analysis_results/drfs_even_time_years_dist.png')

#dist age
plt.hist('age', bins=100, data=cd, color='red')
plt.grid(True)
plt.xlabel('AGE (years)')
plt.ylabel('Count')
plt.title('AGE destribution')
plt.savefig('cd_analysis_results/age_years_dist.png')

for col in cd.select_dtypes('float'):
    plt.figure()
    sns.distplot(R_df[col], label='R', bins=50)
    sns.distplot(NR_df[col], label='NR', bins=50)
    plt.ylabel('Count')
    plt.grid(True)
    plt.title('Distribution of {}'.format(col))
    plt.legend()
    plt.savefig('cd_analysis_results/{}.png'.format(col))
    
plt.figure(figsize=(20,10))
sns.countplot(x='age', hue='RECIST', data=cd)
plt.grid(True)
plt.savefig('cd_analysis_results/age_based_response.png')


#pearson correlation 
#pearson correlation 
for col in cd.select_dtypes('object'):
    plt.figure(figsize=(3,3))
    sns.heatmap(pd.crosstab(cd['RECIST'], cd[col], normalize=True), annot=True, fmt="f", cmap='YlGnBu')
    plt.savefig('cd_analysis_results/{}_corr.png'.format(object))
    
#pearson correlation 
sns.heatmap(cd.corr(), annot=True)
plt.savefig('cd_analysis_results/corr_heatmap.png')

####################################################### DATAFRAMES ##############################
dist_obj = pd.DataFrame()
for objet in cd.select_dtypes('object'):
    dist_obj[objet] = cd[objet].value_counts() 
dist_obj.to_csv('cd_analysis_results/object_distribution.csv')

stat_dfs_age = pd.DataFrame(index=['min', 'max', 'std'])
l_dsf = []
l_dsf.append(cd['drfs_even_time_years'].min())
l_dsf.append(cd['drfs_even_time_years'].max())
l_dsf.append(cd['drfs_even_time_years'].std())
l_age = []
l_age.append(cd['drfs_even_time_years'].min())
l_age.append(cd['drfs_even_time_years'].max())
l_age.append(cd['drfs_even_time_years'].std())

stat_dfs_age['DFS'] = l_dsf
stat_dfs_age['AGE'] = l_age
stat_dfs_age.to_csv('cd_analysis_results/stat_dfs_age.csv')

#target based on her2
pd.crosstab(cd['her2'], cd['RECIST']).to_csv('cd_analysis_results/target_based_her2.csv')

R_df.to_csv('cd_analysis_results/Reponders_df.csv')
NR_df.to_csv('cd_analysis_results/Non-Reponders_df.csv')
balanced_NR.to_csv('cd_analysis_results/NR_df_balance.csv')

#statistical test
mw_p = []
tt_p = []
for name, col in zip(cd.columns[:-1], cd.drop('RECIST', axis=1)):
    p1 = mannwhitneyu(balanced_NR[col], R_df[col])
    p2 =ttest_ind(balanced_NR[col], R_df[col], equal_var=False)
    mw_p.append(p1)
    tt_p.append(p2) 
d = {'mann_wit': mw_p, 'ttest': tt_p}   
pd.DataFrame(index=[cd.columns[:-1]], data = d).to_csv('cd_analysis_results/stat_test_df.csv')

cd.to_csv('cd_analysis_results/final_clinical_data.csv')












































