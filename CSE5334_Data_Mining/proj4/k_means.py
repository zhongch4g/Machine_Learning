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
        # @ Cluster don't need label
        # label = data[:, -1]
        # label = self.oneHotEncoding(label)
        return feature


    def k_means(self, data, k):
        # @ Randomly select k node as center
        train_sample = random.sample(range(len(data)), k)
        center_node = data[train_sample]
        k_part = {}
        Iteration_time = 1
        while 1:
            print("Iteration_time: ", Iteration_time)
            print("Center Node: \n", center_node)
            for i in range(k):
                k_part[i] = []
            for node in data:
                min_distance_idx = np.argmin(np.sqrt(np.sum(np.square(center_node - node.T), axis=1)))
                k_part[min_distance_idx].append(node.tolist())

            # @ Compute center node, and observe it change or not.
            new_center_node = []
            for i in range(k):
                center = np.mean(np.array(k_part[i]), axis=0).tolist()
                new_center_node.append(center)

            new_center_node = np.array(new_center_node)

            if (new_center_node == center_node).all():
                print("Center node no longer change. Cluster Finished...")
                print("Cluster result: ")
                for k, v in k_part.items():
                    print("Class ", k, " contains ", len(v), "nodes...")
                    print(v)
                break
            else:
                center_node = new_center_node
            Iteration_time += 1


    def run(self):
        feature = self.load_csv("Iris.csv", ",")
        self.k_means(feature, k=3)


kmeans = KMeans()
kmeans.run()