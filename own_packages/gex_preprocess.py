
#HANDLING AND PLOTING
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#warning avoid
import warnings
warnings.filterwarnings("ignore")

#SCALING 
from sklearn.preprocessing import StandardScaler, MinMaxScaler, PolynomialFeatures


#import genetic data
gex = pd.read_csv('datas/05_hatzis_2011_gex1_commun_c07.csv', index_col=0)
cd = pd.read_csv('cd_analysis_results/final_clinical_data.csv')


##################################### function initialisation ########################
scaler = MinMaxScaler()

###################################### HANDLING #############################################
#TRANSFORMERS genes expression into scaled data
gex = pd.DataFrame(scaler.fit_transform(gex.values), columns=gex.columns, index=gex.index)
gex.reset_index(inplace=True)

#df.drop(['age','her2'], axis=1, inplace=True)
cd.drop(['age','pr_ihc'], axis=1, inplace=True)
gex.drop(['KLK3'], axis=1, inplace=True)

#index reset (to merge)
cd = cd.reset_index().rename(columns={'patient_ID':'Patient'})
gex.set_index('Patient', inplace=True)

########################################## DATAFRAME ##################################
df = pd.merge(cd, gex, on='Patient')
df.to_csv('gex_preprocess_results/main_data_sc.csv',index=0)

######################################### PLOTING #########################################
#distribution of 10 first gene expression 
plt.figure(figsize=(5,5))
for col in gex.iloc[:,1:10]:
    sns.distplot(gex[col], label=col, kde_kws={'lw':3})
    plt.grid(True)
    plt.xlabel('Expression values')
    plt.title('Genes expression distribution (10 ramdom genes)')
    plt.legend()
plt.savefig('gex_preprocess_results/genes_exp_dist.png')

######################################### PRINTERS ############################################
print('gene expression shape  :',gex.shape)
print('clinical shape',cd.shape)
print('merge_shape: ',df.shape)


    
