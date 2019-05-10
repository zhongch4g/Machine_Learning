#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/15 7:01 PM
# @Author  : zhongch4g
# @Site    : 
# @File    : naive_bayes.py
# @Software: IntelliJ IDEA

import os
import random
import math
import re
import time
import sys


class NaiveBayes(object):
    def __init__(self):
        self.root = "./smsspamcollection/"
        self.data_set = "SMSSpamCollection"
        self.stop_words_path = "./stop_words_eng.txt"
        # Filter some words， like 'the'、'is'、'at'、'which'、'on', don't have actual meaning. Reduce space & improve efficiency
        self.stop_words = self.load_stop_words()

    def load_stop_words(self):
        print("Loading stop words...")
        # Stop words
        stop_words = []
        with open(self.stop_words_path, 'r') as r:
            stop_words = r.read().split('\n')
        print("Loading stop words successfully...")
        return stop_words

    def load_data(self):
        print("Loading data...")

        # @ Each doc that belongs to class
        data_X, data_y = [], []
        with open(self.root + self.data_set, 'r', encoding="ISO8859") as r:
            for line in r.readlines():
                if line == '\n':
                    continue

                # line_words = re.split(r"[*:,\-.\s()<>;!?\[\]\'\"]", line)
                line_words = line.split("\t")
                if len(line_words) == 2:
                    # @ Sentence
                    data_X.append(line_words[1])
                    # @ label
                    data_y.append(line_words[0].lower())
                else:
                    data_X.append(" ".join([line_words[i] for i in range(1, len(line_words))]))
                    data_y.append(line_words[0].lower())

        # word_bags model
        data_word_bags_X = []
        for sentence in data_X:
            data_word_bags_X.append(self.split_data(sentence))

        print("Load data successfully...")
        print("Spam number: ", data_y.count("spam"), "Ham number: ", data_y.count("ham"))
        return data_word_bags_X, data_y

    # split sentence into a words, and return word count dictionary
    def split_data(self, sentence):
        word_bags = {}
        for word in re.split(r"[*:,\-.\s()<>;!?\[\]\'\"/]", sentence):
            # @ Add stop words
            if word == "" and word in self.stop_words:
                continue
            word_bags.setdefault(word.lower(), 0)
            word_bags[word.lower()] += 1
        return word_bags

    def predict(self, train_wb_spam, train_wb_ham, document):
        max_prob, pre_class = -sys.maxsize-1, None
        count_words = sum(train_wb_ham.values()) + sum(train_wb_ham.values())
        train_words = len(set(list(train_wb_spam.keys())+list(train_wb_ham.keys())))
        for cls in ["spam", "ham"]:
            if cls == "ham":
                # @ whether is belongs to spam
                prior = self.num_ham / (self.num_spam + self.num_ham)
                condition = 1
                posterior = 0
                for word, cnt in document.items():
                    train_wb_ham.setdefault(word, 0)
                    condition += math.log((train_wb_ham[word] + 1)/(count_words + train_words))*cnt
                posterior = condition * prior

            if cls == "spam":
                # @ whether is belongs to spam
                prior = self.num_spam / self.num_spam + self.num_ham
                condition = 1
                posterior = 0
                for word, cnt in document.items():
                    train_wb_spam.setdefault(word, 0)
                    condition += math.log((train_wb_spam[word] + 1)/(count_words + train_words))*cnt
                posterior = condition * prior

            if max_prob < posterior:
                max_prob = posterior
                pre_class = cls
        return pre_class

    def naive_bayes(self, train_wb_spam, train_wb_ham, test_X, test_y):
        correct = 0
        count = 0
        eva = 0
        TP, FP, FN, TN = 0, 0, 0, 0
        for doc in range(len(test_X)):
            count += 1
            # @doc is composed by a dict with words count
            pre_class = self.predict(train_wb_spam, train_wb_ham, test_X[doc])
            if test_y[doc] == "ham" and pre_class == test_y[doc]:
                TP += 1
            if test_y[doc] == "ham" and pre_class != test_y[doc]:
                FN += 1
            if test_y[doc] == "spam" and pre_class == test_y[doc]:
                TN += 1
            if test_y[doc] == "spam" and pre_class != test_y[doc]:
                FP += 1
        recall = TP / (TP + FN)
        precision = TP / (TP + FP)
        accuracy = (TP + TN) / ( TP + TN + FP + FN)
        f1 = 2 / ((1 / recall) + (1 / precision))

        print("Recall: ", recall, "Precision: ", precision, "Accuracy: ", accuracy, "f1: ", f1)

    # K fold cross-validation
    def k_fold(self, data_X, data_y, k):
        for select in range(k):
            # Select data for each iteration
            train_X, train_y, test_X, test_y = \
                self.select_n_as_test(data_X, data_y, select, k)

            # @ Generate train set as a big word bags, divide into spam and ham
            train_wb_spam, train_wb_ham = self.big_word_bags(train_X, train_y)
            self.naive_bayes(train_wb_spam, train_wb_ham, test_X, test_y)

    # bag of words
    def big_word_bags(self, train_X, train_y):
        train_wb_spam, train_wb_ham = {}, {}
        num_spam, num_ham = 0, 0
        for i in range(len(train_X)):
            if train_y[i] == "spam":
                num_spam += 1
                for k, v in train_X[i].items():
                    train_wb_spam.setdefault(k, 0)
                    train_wb_spam[k] += v
            else:
                num_ham += 1
                for k, v in train_X[i].items():
                    train_wb_ham.setdefault(k, 0)
                    train_wb_ham[k] += v
        self.num_spam, self.num_ham = num_spam, num_ham
        return train_wb_spam, train_wb_ham

    # Choose train set and test test, n-th part as test set
    def select_n_as_test(self, data_X, data_y, n, k):
        divide_index = []
        train_X, train_y, test_X, test_y = [], [], [], []
        # @ divide index
        sep = len(data_X) // k
        temp = []
        for i in range(len(data_X) + 1):
            if i % sep != 0:
                temp.append(i)
            else:
                if i != 0:
                    divide_index.append(temp)
                    temp = []
                else:
                    temp.append(i)
        for i in range(k):
            if i == n:
                for j in divide_index[i]:
                    test_X.append(data_X[j])
                    test_y.append(data_y[j])
            else:
                for k in divide_index[i]:
                    train_X.append(data_X[k])
                    train_y.append(data_y[k])
        return train_X, train_y, test_X, test_y

    def run(self):
        t1 = time.time()
        data_X, data_y = self.load_data()

        self.k_fold(data_X, data_y, k = 10)

        # self.naive_bayes(self.train, self.test)
        print("Time cost: ", time.time() - t1)


"""
Steps:
1、Read data from UCI Repository (SMS Spam Collection Data Set) https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection
2、Read stop words (Download from internet)
3、Data pre-processing
4、Construct naïve bayes model
5、Evaluate the model by using k fold cross-validation
"""
nb = NaiveBayes()
nb.run()
