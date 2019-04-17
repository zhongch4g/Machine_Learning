#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/16 10:07 PM
# @Author  : zhongch4g
# @Site    : 
# @File    : k_means.py
# @Software: IntelliJ IDEA

import numpy as np
import random


class KMeans(object):
    def __init__(self):
        pass

    def load_csv(self, path, delimiter):
        # load data
        print("loading data...\n")
        data = np.loadtxt(path, dtype=np.str, delimiter=delimiter)

        # set feature from str to float64
        feature = data[:, :-1].astype(np.float64)

        label = data[:, -1]

        label = self.oneHotEncoding(label)
        return feature, label

    def k_means(self, data, k):
        # @ Randomly select k node as center
        train_sample = random.sample(range(len(data)), k)



    def run(self):
        feature, label = self.load_csv("Iris.csv", ",")
        self.k_means(feature, k=3)


kmeans = KMeans()
kmeans.run()