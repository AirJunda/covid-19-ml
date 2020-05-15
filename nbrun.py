# -*- coding: utf-8 -*-
"""
Created on Thu Apr 23 03:16:13 2020

@author: Air Junda
"""

from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier
import preprocessing as pp






def crossVal(clf,x_data, y_data,n=5):
    "Cross validation"
    result = []
    f1s = []
    for _ in range(n):
        x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
        clf.fit(x_train, np.ravel(y_train))
        y_pred = clf.predict(x_test)
        accu = accuracy_score(y_test, y_pred)
        f1score = f1_score(y_test, y_pred, average='macro')
        result.append(accu)
        f1s.append(f1score)
    meanacc = np.mean(result)
    meanf1 = np.mean(f1s)
    
    std1 = np.std(result) 
    std2 = np.std(f1s)
    return (meanacc,std1, meanf1,std2)
    
    

def baseline(x_data, y_data, stra = "uniform"):
    """ baseline prediction using dummyClassifier """
    x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2)
    dummy = DummyClassifier(strategy= stra)
    dummy.fit(x_train, y_train)
    y_pred = dummy.predict(x_test)
    accu = accuracy_score(y_test, y_pred)
    return accu



def pickledf(df ,filepath):
    "persistence storage"
    df.to_pickle(filepath)
    
    
def loaddf(filepath):
    "Load picked dataframe from local dir"
    # dir = "pickle/df1.pkl" for example
    unpickled_df = pd.read_pickle(filepath)
    return unpickled_df

def NaNfilling(df):
    # filling with -99
    for column in df.columns:
        df[column] = df[column].fillna(0)
            
    
    
def search(x_data, y_data, n = 5):
    """Apply grid-search for best alpha value"""
    alpha = np.arange(0.01, 8, 0.01)
    param_grid = {'alpha' : alpha}                
    clf = MultinomialNB()  
    grid_search = GridSearchCV(clf, param_grid, cv=n)
    grid_search.fit(x_data, y_data)
    return grid_search.best_params_
            
    
if __name__ == "__main__":
    "If pickle/df7.pkl does not exist, uncomment the code below"
    #pp.initial()
    df7 = loaddf(filepath='pickle/df7.pkl')
    # NaN handle
    df7.dropna(axis=1, how='any', thresh=9000, inplace=True)
    X = df7.iloc[:,0:-1]
    y = df7['target']
    
    NaNfilling(X)
    nulldists = X.isna().sum()  # check if all null values is filled by visual examination of nulldist
    # ready for clasification
    print('ready for run benchmark')
    model = MultinomialNB()
    bsl_rs = baseline(X,y, stra ="most_frequent")  # baseline accuracy
    nb_benchm = crossVal(model,X, y,n=10)          # naive bayes result 
    
    # paramter tunning
    print('Model tuning in progress')
    bestparms = search(X, y, n = 2)
    clfbest = MultinomialNB(alpha = bestparms['alpha'])
    bestreport = crossVal(clfbest,X, y,10)

    print("The best alpha value is " + str(round(bestparms['alpha'],4)))
    print("The accuracy of the NB with best alpha value is " + str(round(bestreport[0],4)))
    print("The f1score of the NB with best alpha value is " + str(round(bestreport[2],4)))
    


