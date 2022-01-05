from flask import Flask, render_template, request
import pandas as pd
from flask.wrappers import Request
from flask_sqlalchemy import SQLAlchemy
import glob
import os
from analysis import csv_hundling, check_data, count_value, plot_pie, float_dist, pearson_corr
from own_packages.functions import preprocess, evaluation_skf
from mocks import Client
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

##################################################################################Models#######################################################3
seed = 0

cart_clf = DecisionTreeClassifier(random_state=seed)
knn_clf = KNeighborsClassifier(n_neighbors = 5,  n_jobs = 1)
#xgb_clf = XGBClassifier(random_state=seed)
svcl_clf = SVC(random_state=seed, probability=True, kernel='linear') #decision function
lr_clf = LogisticRegression(random_state =seed, dual = False, class_weight = None,  n_jobs = 1) 

MLA = [cart_clf,
       knn_clf,
       svcl_clf,
       lr_clf]

skf = StratifiedKFold(n_splits=10)
repitition = np.random.randint(0, 99999999, size = 5)  

##################################################################################Models#######################################################3

app = Flask(__name__)

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///database/db.sqlite3'
# desactiver le warning en question 
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db = SQLAlchemy(app)

#sqlite prend cette classe en concideration afin de nous cree une table
#creation du model (fait office de la base de donnee)
class Disc(db.Model):
    __tablename__ = 'Dicovery_data'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    content = db.Column(db.Text)
    def __repr__(self):
        return '<row {}, {}>'.format(self.id, self.title)

class Indep(db.Model):
    __tablename__ = 'Independent_data'
    id = db.Column(db.Integer, primary_key=True)
    title = db.Column(db.String(255))
    content = db.Column(db.Text)
    def __repr__(self):
        return '<row {}, {}>'.format(self.id, self.title)



#la fonction permet de retourner un template (une page html)
@app.route('/', methods=['GET', 'POST'])
def home():
    return render_template('pages/home/index.html')

@app.route('/discovery_dataset')
def discovery_dataset():
    disc_data = Disc.query.all()
    return render_template('pages/discovery_dataset/index.html', disc_data = disc_data)

@app.route('/discovery_dataset/<int:id>')
def discovery_dataset_show(id): 
    row_data = Disc.query.get(id)
    return render_template('pages/discovery_dataset/show.html', row_data=row_data)

@app.route('/independent_dataset')
def independent_dataset():
    indep_data = Indep.query.all()
    return render_template('pages/independent_dataset/index.html', indep_data = indep_data)

@app.route('/independent_dataset/<int:id>')
def independent_dataset_show(id): 
    row_data = Indep.query.get(id)
    return render_template('pages/independent_dataset/show.html', row_data=row_data)

@app.route('/data_comparison')
def data_comparison():
    data = pd.read_csv('comparison.csv', encoding='latin1')
    return render_template('pages/data_comparison/index.html', data=data.to_html())



@app.route('/descriptive_analysis',  methods=['GET', 'POST'])
def descriptive_analysis():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = csv_hundling(file) #load & dropna
        check = check_data(data) #shape / type / nan rate
        count = count_value(data) #list of count value
        object_name = plot_pie(data) #dist of object var
        float_name = float_dist(data) # dit of float var
        data = data.head(10) #display 10 rows only
        pearson_corr(data)
        return render_template('pages/descriptive_analysis/index.html',
        data=data.to_html(),
        check=check, 
        count=count,
        float_name=float_name,
        object_name=object_name)
    
@app.route('/classification_jnb')
def classification_jnb():
    return render_template('pages/classification_jnb/index.html')

@app.route('/classification_jnb/default_models')
def default_models():
    return render_template('pages/classification_jnb/default_models/index.html')


@app.route('/classification_jnb/default_models',  methods=['GET', 'POST'])
def result_models():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = csv_hundling(file) #load & dropna
        X, y = preprocess(data)
        default_models = pd.DataFrame(columns=['MCC_SCORE', 'AUC_SCORE', 'ACC_SCORE'])     
        for model in MLA:
            print(model.__class__.__name__)
            for seed in repitition:
                print('seed number : ',seed)
                default_models.loc['{} | {} '.format(model.__class__.__name__,seed)] = evaluation_skf(model, X, y, skf)
                print('_'*50)
                default_models.to_csv('scores_result/default_models.csv') 

        
        data = data.head(10) #display 10 rows only
        return render_template('pages/classification_jnb/default_models/result.html', data=data.to_html(), default_models=default_models.to_html())

'''@app.route('/classification_jnb/default_models/default_results',  methods=['GET', 'POST'])
def default_result():
    if request.method == 'POST':
        file = request.form['upload-file']
        data = csv_hundling(file) #load & dropna
        data = data.head(10) #display 10 rows only
        X, y = preprocess(data)
        default_models = pd.DataFrame(columns=['MCC_SCORE', 'AUC_SCORE', 'ACC_SCORE'])     
        for model in MLA:
            print(model.__class__.__name__)
            for seed in repitition:
                print('seed number : ',seed)
                default_models.loc['{} | {} '.format(model.__class__.__name__,seed)] = evaluation_skf(model, X, y, skf)
                print('_'*50)
                default_models.to_csv('pages/classification_jnb/default_models/default_results/default_models.csv') 

        return render_template('pages/classification_jnb/default_models/result.html', data=data, default_models=default_models.to_html())'''


@app.route('/classification_pipeline')
def classification_pipeline():
    #clients = Client.all()
    return render_template('pages/classification_pipeline/index.html')


@app.errorhandler(404)
def page_not_found(error):
    return render_template('errors/404.html'), 404

# si j'execute le fichier flaskproject.py, cela va lancer mon application 
# debug = True ==> je suis en developpement, cela relance le debgeur apres chaque modification
# cree la base de donne si elle n'exciste pas 
if __name__=='__main__':
    db.create_all()
    app.run(debug=True, port=3000)
