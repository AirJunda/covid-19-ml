# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:12:53 2020

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
    kval = np.arange(3, 20, 1)
    param_grid = {'n_neighbors' : kval}                
    clf = KNeighborsClassifier()
    grid_search = GridSearchCV(clf, param_grid, cv=n)
    grid_search.fit(x_data, y_data)
    return grid_search.best_params_


def predictCOVID(processed_cov):
    # predict the oriign of COVID-19 
    # choose the animal with max value in RES
    cov = processed_cov
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2,shuffle)
    
    mc = X_train.columns.difference(cov.columns)
    for newcol in mc: cov[newcol]= 0
    rf = cov.columns.difference(X_train.columns)
    cov = cov.drop(rf, axis=1)
    cov = cov[X_train.columns]
    pred = model.predict(cov)
    camel = 0
    bat = 0
    chicken = 0
    monkey = 0
    pangolin = 0
    for i in range(len(pred)):
        if pred[i] == 'camel':
            camel += 1
        if pred[i] == 'chicken':
            chicken += 1
        if pred[i] == 'bat':
            bat += 1
        if pred[i] == 'pangolin':
            pangolin += 1
        if pred[i] == 'monkey':
            monkey += 1
    RES = {'camel': camel, 'bat':bat, 'chicken':chicken, 'pangolin':pangolin, 'monkey':monkey}
    return RES
            
    
if __name__ == "__main__":
    "If pickle/df7.pkl does not exist, uncomment the code below"
    #pp.initial()
    df7 = loaddf(filepath='pickle/df7.pkl')
    # NaN handle
    df7.dropna(axis=1, how='any', thresh=9000, inplace=True)
    X = df7.iloc[:,0:-1]
    y = df7['target'] 
    NaNfilling(X)
    null_distribution = X.isna().sum()  # check if all null values is filled by visual examination of nulldist
    # ready for clasification
    print('Using KNN to train')
    model = KNeighborsClassifier(n_neighbors=5)
    knn_benchm = crossVal(model,X, y,n=2)  
    
    
    print("The accuracy of the KNN with best alpha value is " + str(round(knn_benchm[0],4)))
    print("The f1score of the KNN with best alpha value is " + str(round(knn_benchm[2],4)))
    
    cov = pp.process_file('human.fasta','COVID-19')
    cov = cov.drop('target', axis=1)
    predict_covid19 = predictCOVID(cov)
    origin = max(predict_covid19, key=predict_covid19.get) 
    print('The origin of COVID-19 is: ' +origin)  
    
    


