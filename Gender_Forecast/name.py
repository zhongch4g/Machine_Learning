#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
"""
__author__ = 'sunzhe3'

import time
import os
import sys

import numpy as np
import pandas as pd
import re

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.externals import joblib
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import classification_report
from sklearn.ensemble import GradientBoostingClassifier


def savesklearnobject(var, filename='', path=''):
    filename = re.sub('\W', '', str(type(var)).split('.')[-1].lower()) + '_' + filename + '.plk'
    if path == '':
        path = os.path.join(os.getcwd())
    joblib.dump(var, path + os.path.sep + filename)


def readsklearnobject(filepath):
    return joblib.load(filepath)


def main():
    print("------------main--begin----------------")
    input_file1 =  "name1.csv"
    input_file2 =  "br21.csv"

    s = time.clock()
    name_df = pd.read_csv(input_file1, header=None, sep=',', names=['pin', 'n21', 'gender'], index_col='pin')
    br2_df = pd.read_csv(input_file2, header=None, sep=',', names=['pin', 'cate2', 'pv', 'tm'],
                         usecols=['pin', 'cate2', 'pv'])
    print('read data takes: ' + str(round(time.clock() - s, 3)) + ' s')

    s = time.clock()
    # 品类在内存中  行转列  转以后缺一个品类和特征位置的对应关系 预测的时候会有问题
    br2_train = pd.pivot_table(br2_df, values=['pv'], index=['pin'], columns=['cate2'], fill_value=0, aggfunc=np.sum)


    # 多级索引拉平
    br2_train.columns = ['_'.join(col) for col in br2_train.columns]
    br2_train = pd.merge(br2_train, name_df.ix[:, 'gender'].to_frame(), how='inner', left_index=True, right_index=True)
    br2_train_x = br2_train.filter(regex="pv_.*")
    print("br2 train x memory usage :"+str(br2_train_x.memory_usage().sum()/1024/1024)+" MB")
    br2_train_y = br2_train['gender']


    # Tfidf 
    tfidf_vec = TfidfVectorizer(binary=False, decode_error='ignore')

    # ? 将pin和姓名放一块
    name_train_x = tfidf_vec.fit_transform(name_df.ix[:, 'n21'])
    name_train_y = name_df['gender']
    print("name feature number: " + str(len(tfidf_vec.get_feature_names())))
    savesklearnobject(tfidf_vec, filename='name_1')
    print('preprocess data takes: ' + str(round(time.clock() - s, 3)) + ' s')

    s = time.clock()
    x_train, x_test, y_train, y_test = train_test_split(name_train_x, name_train_y, test_size=0.3)

    name_clfr = BernoulliNB().fit(x_train, y_train)
    predict_y = name_clfr.predict(x_test)
    predict_train_y = name_clfr.predict(x_train)
    savesklearnobject(name_clfr, filename='name_1')
    print('name train result : \n' + classification_report(y_train, predict_train_y))
    print('name test result : \n' + classification_report(y_test, predict_y))
    print('name modle train takes: ' + str(round(time.clock() - s, 3)) + ' s')

    s = time.clock()
    x_train, x_test, y_train, y_test = train_test_split(br2_train_x, br2_train_y, test_size=0.3)
    br2_clfr = GradientBoostingClassifier(n_estimators=200, learning_rate=0.1, max_depth=5, random_state=42)
    br2_clfr.fit(x_train, y_train)
    predict_y = br2_clfr.predict(x_test)
    # print(br2_clfr.predict_proba([4,5,0,0,0,0,0,0,0,0,13,10,0,0,3,1]))
    predict_train_y = br2_clfr.predict(x_train)
    savesklearnobject(br2_clfr, filename='name_1')
    print('br2 train result : \n' + classification_report(y_train, predict_train_y))
    print('br2 test result : \n' + classification_report(y_test, predict_y))
    print('br2 modle train takes: ' + str(round(time.clock() - s, 3)) + ' s')

    s = time.clock()
    name_train = pd.DataFrame(data=name_clfr.predict_proba(name_train_x), columns=name_clfr.classes_, index=name_train_y.index)
    pv_train = pd.DataFrame(data=br2_clfr.predict_proba(br2_train_x), columns=br2_clfr.classes_, index=br2_train_x.index)
    pv_name_train_x = pd.merge(name_train, pv_train, how='inner', left_index=True, right_index=True, suffixes=['_name','_pv']).filter(regex="1_.*")
    pv_name_train_y = br2_train_y
    x_train, x_test, y_train, y_test = train_test_split(pv_name_train_x, pv_name_train_y, test_size=0.3)
    br2_name_clfr = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    br2_name_clfr.fit(x_train, y_train)
    predict_y = br2_name_clfr.predict(x_test)
    predict_train_y = br2_name_clfr.predict(x_train)
    savesklearnobject(br2_name_clfr, filename='name_2')
    print('br2 and name train result : \n' + classification_report(y_train, predict_train_y))
    print('br2 and name test result : \n' + classification_report(y_test, predict_y))
    print('br2 and name modle train takes: ' + str(round(time.clock() - s, 3)) + ' s')


if __name__ == '__main__':
    main()
