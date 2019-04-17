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
        """
        self.train = {class1:{500 docs words count}, class2:{} ...}
        self.test = {class1:[{1 doc words count}, {1 doc words count}, class2:[{}, {}, {}]]}
        """
        self.root = "./20_newsgroups/"
        self.classes = ["alt.atheism", "comp.graphics", "comp.os.ms-windows.misc", "comp.sys.ibm.pc.hardware"
            , "comp.sys.mac.hardware", "comp.windows.x", "misc.forsale", "rec.autos", "rec.motorcycles"
            , "rec.sport.baseball", "rec.sport.hockey", "sci.crypt", "sci.electronics", "sci.med"
            , "sci.space", "soc.religion.christian", "talk.politics.guns", "talk.politics.mideast"
            , "talk.politics.misc", "talk.religion.misc"]
        self.all_train_words = set()
        self.train, self.test = self.load_data()

    def load_data(self):
        print("Loading data...")
        # @ Each class include 1000 documents, we set 500 documents as train set.

        document_list = []
        for cls in self.classes:
            document = []
            for filename in os.listdir(self.root+cls):
                document.append(self.root+cls+"/"+filename)
            document_list.append(document)
        print(len(document_list), len(document_list[0]))

        # @ Read each document
        train_doc_word_bags = {}
        test_doc_word_bags = {}
        for cls in self.classes:
            train_doc_word_bags.setdefault(cls, )
            test_doc_word_bags.setdefault(cls, [])

    # Stop words
        stop_words = []
        with open("./stop_words_eng.txt", 'r') as r:
            stop_words = r.read().split('\n')

        # @ Randomly select 500 sample
        #
        train_sample = random.sample(range(1000), 500)
        test_sample = [i for i in range(1000) if i not in train_sample]
        # @ Each class
        # @ document_list : [[], [], []] ==> class_path: class_document
        for idx, cls_path in enumerate(document_list):
            train_doc_words = {}
            # @ Each doc that belongs to class
            for idx2, doc in enumerate(cls_path):
                test_doc_words = {}
                # @ Generate train set
                if idx2 in train_sample:
                    doc_head_flag = True
                    with open(doc, 'r', encoding="ISO8859") as r:
                        for line in r.readlines():
                            if line == '\n' and doc_head_flag is False:
                                continue
                            if line == '\n' and doc_head_flag is True:
                                doc_head_flag = False
                            if doc_head_flag:
                                continue
                            else:
                                if line == '\n':
                                    continue
                                line_words = re.split(r"[*:,\-.\s()<>;!?\[\]\'\"]", line)
                                for word in line_words:
                                    word = word.lower()
                                    # @ Normalization & stop words filter
                                    if word not in stop_words and word != "" and 'a' <= word[0] <= 'z':
                                        self.all_train_words.add(word)
                                        train_doc_words.setdefault(word, 0)
                                        train_doc_words[word] += 1
                if idx2 in test_sample:
                    # @ Generate test set
                    doc_head_flag = True
                    with open(doc, 'r', encoding="ISO8859") as r:
                        for line in r.readlines():
                            if line == '\n' and doc_head_flag is False:
                                continue
                            if line == '\n' and doc_head_flag is True:
                                doc_head_flag = False
                            if doc_head_flag:
                                continue
                            else:
                                if line == '\n':
                                    continue
                                line_words = re.split(r"[:,\-.\s()<>;!?\[\]\'\"]", line)
                                for word in line_words:
                                    word = word.lower()
                                    # @ Normalization & stop words filter
                                    if word not in stop_words and word != "" and 'a' <= word[0] <= 'z':
                                        test_doc_words.setdefault(word, 0)
                                        test_doc_words[word] += 1
                    if test_doc_words != {}:
                        test_doc_word_bags[self.classes[idx]].append(test_doc_words)
            train_doc_words = {k:v for k, v in train_doc_words.items() if v > 0}
            train_doc_word_bags[self.classes[idx]] = train_doc_words
        print("Load data successfully...")
        return train_doc_word_bags, test_doc_word_bags

    def predict(self, train, doc):
        max_prob, pre_class = -sys.maxsize-1, None
        for cls in train.keys():
            # @ Count the number of words in cls
            count_words = 0
            for n in train[cls].values():
                count_words += n
            # print("count_words", count_words)

            # @ whether is belongs to cls
            prior = 1.0 / 20
            condition = 1
            posterior = 0
            for word, cnt in doc.items():
                train[cls].setdefault(word, 0)
                condition += math.log((train[cls][word] + 1)/(count_words + len(self.all_train_words)))*cnt
            posterior = condition * prior
            if max_prob < posterior:
                max_prob = posterior
                pre_class = cls

        return pre_class

    def naive_bayes(self, train, test):
        correct = 0
        count = 0
        eva = 0
        for cls, docs in test.items():
            print("docs count: ", count)
            for doc in test[cls]:
                count += 1
                # @doc is composed by a dict with words count
                pre_class = self.predict(train, doc)
                if pre_class == cls:
                    correct += 1
            print("Current correct: ", correct,"Current number: ", count)
        print("Accuracy :", correct / count)

    def run(self):
        t1 = time.time()
        self.naive_bayes(self.train, self.test)
        print(time.time() - t1)

nb = NaiveBayes()
nb.run()
