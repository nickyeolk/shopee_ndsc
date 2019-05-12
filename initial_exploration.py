#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split





def mapk(y_allpred, y_actual, k):
    '''Calculates the mean average precision at a threshold of k.'''
    assert len(y_allpred)==len(y_actual), 'Lengths are of pred and actual are different'
    scores=[]
    for jj, y_pred in enumerate(y_allpred):
        if(len(y_pred)==1):
            scores.append(apk([y_pred], [y_actual[jj]], k))
        else: scores.append(apk(y_pred, [y_actual[jj]], k))
    return np.mean(scores)

def apk(y_pred, y_actual, k):
    '''Calculates average precision for a sequence of predictions'''
    score=0
    count=0
    for ii in range(min(k, len(y_pred))):
        if ((y_pred[ii] in y_actual) and (y_pred[ii] not in y_pred[:ii])):
            count+=1
            score=score+count/(ii+1)
    return score/min(len(y_actual), k)
            

def validate_accuracy(df, item, tfidf, f):
    X_train, X_test, y_train, y_test = train_test_split(tfidf, df[item], test_size=0.2, random_state=0)
    lr=LogisticRegression(solver='lbfgs', multi_class='ovr', n_jobs=-1)
    lr.fit(X_train,y_train)
    allprobs=lr.predict_proba(X_test)
    sortprobs=np.argsort(allprobs, axis=1)[:, ::-1]
    classes=lr.classes_
    y_predicts=classes[sortprobs]
    stringg='{} {} \n'.format(item, mapk(y_predicts.tolist(), list(y_test), 1))
    f.write(stringg)

f= open("test_results.txt","a+")

df_fash=pd.read_csv('./data_raw/fashion_data_info_train_competition.csv')
fashion_items=['Pattern', 'Collar Type', 'Sleeves', 'Fashion Trend', 'Clothing Material']
df_fash=df_fash.fillna(-1)

vectorizer = CountVectorizer(analyzer = "word", strip_accents=None, tokenizer = None, preprocessor = None,                              stop_words = None, max_features = 5000, ngram_range=(1,3)) 
train_data_features = vectorizer.fit_transform(df_fash['title'])
tfidfier = TfidfTransformer()
tfidf = tfidfier.fit_transform(train_data_features)
for fashion_item in fashion_items:
    validate_accuracy(df_fash, fashion_item, tfidf, f)



df_beaut=pd.read_csv('./data_raw/beauty_data_info_train_competition.csv')
df_beaut.info()
df_beaut=df_beaut.fillna(-1)
beauty_items=[thing for thing in df_beaut.columns if thing not in ['itemid', 'image_path', 'title']]
print(beauty_items)
vectorizer = CountVectorizer(analyzer = "word", strip_accents=None, tokenizer = None, preprocessor = None,                              stop_words = None, max_features = 5000, ngram_range=(1,3))
train_data_features = vectorizer.fit_transform(df_beaut['title'])
tfidfier = TfidfTransformer()
tfidf = tfidfier.fit_transform(train_data_features)

for beauty_item in beauty_items:
    validate_accuracy(df_beaut, beauty_item, tfidf, f)





df_mobile=pd.read_csv('./data_raw/mobile_data_info_train_competition.csv')
df_mobile.info()
df_mobile=df_mobile.fillna(-1)
mobile_items=[thing for thing in df_mobile.columns if thing not in ['itemid', 'image_path', 'title']]
print(mobile_items)
train_data_features = vectorizer.fit_transform(df_mobile['title'])
tfidfier = TfidfTransformer()
tfidf = tfidfier.fit_transform(train_data_features)
for mobile_item in mobile_items:
    validate_accuracy(df_mobile, mobile_item, tfidf, f)


f.close()
