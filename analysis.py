from flask import Flask, render_template, request
import pandas as pd
from flask.wrappers import Request
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#STATISTIC TEST
from scipy.stats import ttest_ind, mannwhitneyu

#warning avoid
import warnings
warnings.filterwarnings("ignore")



def csv_hundling(path):
    df = pd.read_csv(path, index_col=0)
    df.dropna(inplace=True)
    return df

def  check_data(df):
    shape = df.shape
    type = df.dtypes
    nan_rate = df.isna().sum().sort_values(ascending=False)
    return shape, type, nan_rate

def count_value(df):
    list = []
    for col in df.select_dtypes('object'): 
        list.append(df[col].value_counts())
    return list

def plot_pie(df):
    #distribution variables categorielles
    list= []
    for object in df.select_dtypes('object'):
        plt.figure()
        with plt.style.context('fivethirtyeight'):
            df[object].value_counts().plot.pie(autopct = lambda x: str(round(x, 2)) + '%', shadow=True)
            plt.title(object)
            plt.savefig('static/images/{}.png'.format(object))
            plt.legend()
            list.append(object)
    return list

def float_dist(df):
    plt.style.context('dark_background')
    list = []
    for col in df.select_dtypes(include=['float64', 'int64']):
        plt.figure()
        with plt.style.context('dark_background'):
            plt.hist(col, bins=100, data=df, color='green')
            plt.ylabel('Count')
            plt.grid(True)
            plt.title('Distribution of {}'.format(col))
            plt.legend()
            plt.savefig('static/images/{}.png'.format(col))
            list.append(col)
    return list

def pearson_corr(df):
    corr = df.corr()
    mask = np.zeros_like(corr)
    mask[np.triu_indices_from(mask)] = True
    with sns.axes_style("ticks"):
        f, ax = plt.subplots(figsize=(7, 5))
        ax = sns.heatmap(corr, mask=mask, vmax=.3, square=True)
    plt.savefig('static/images/heatmap.png')


########################################################################################33






