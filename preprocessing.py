# -*- coding: utf-8 -*-
"""
Created on Thu May 14 21:24:49 2020

@author: Air Junda
"""

from nltk.corpus import stopwords  
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from xgboost import plot_importance
import xgboost# create a train/test split
from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
from sklearn.metrics import accuracy_score

from imblearn.combine import SMOTEENN
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

def generate_ngrams(s1):
    count_vect = CountVectorizer(lowercase=False, ngram_range=(5,5),analyzer='char')
    X1 = count_vect.fit_transform(s1)
    
    lcount = list()
    lcount = []
    cnt_line = 0
    for i in s1:
        print(cnt_line)
        cnt_line += 1
        count = len(i)
        #print(count)
        lcount.append(count)
        count_vect_df = pd.DataFrame(X1.todense(), columns=count_vect.get_feature_names())
    
    count_vect_df=count_vect_df.apply(lambda x: x / lcount[x.name] ,axis=1)
        
    return count_vect_df


def process_file(filename,target_val):
    f = open(filename) #'datasets\\corona-nucleo-chicken-complete.fasta')
    lines = ""
    s1 = list()
    step = 0
    term = 0
    for line in f:
        line = ''.join(line.split())
        if line.startswith(">") and step==0:
            line = line.split('>',1)[0].strip()
            step = step + 1
        if line.startswith(">") and step >= 1:
            line = line.split('>',1)[0].strip()
            s1.append(lines)
            lines = ""
            step = step + 1
            term = 0
        lines = lines + line
    count_vect_df = generate_ngrams(s1) 
    count_vect_df['target'] = target_val
    return count_vect_df


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


def NaNfilling(df):
    # filling with -99
    for column in df.columns:
        df[column] = df[column].fillna(0)
            


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
            
def initial():
    df1 = process_file('chicken.fasta','chicken')
    df2 = process_file('camel.fasta','camel')
    df3 = process_file('pangolin.fasta','pangolin')
    df4 = process_file('bat.fasta','bat')
    df5 = process_file('monkey.fasta','monkey')
    
    df1s = df1
    df1s = df1s.sample(n=4000,random_state=123,axis=0)
    
    df2s = df2
    df2s = df2s.sample(n=4000,random_state=123,axis=0)
    
    tmp = df3
    for _ in range(700):
        df3 = pd.concat([df3, tmp])
        
    tmp = df5
    for _ in range(800):
        df5 = pd.concat([df5, tmp])
        
    df7 = pd.concat([df1s, df2s, df3, df4, df5])
    df7.to_pickle("pickle/df7.pkl")
    
if __name__ == "__main__":
    initial()
    
    
    
    
    