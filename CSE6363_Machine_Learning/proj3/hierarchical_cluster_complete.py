#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 3:02 PM
# @Author  : zhongch4g
# @Site    : 
# @File    : hierarchical_cluster_complete.py
# @Software: IntelliJ IDEA
import sys


"""
@ Implementation of herarchical clustering
@ Using agglomerative clustering ways:
Agglomerative clustering starts with one cluster per data point
and recursively merges clusters according to a cluster similarity measure
@ Linkage Measures:
Single (minimum) linkage: similarity of the two most similar items in the two clusters
n Complete (maximum) linkage: similarity of the two most dissimilar items in the two clusters
"""


class Node(object):
    def __init__(self, val):
        self.val = val
        self.left = None
        self.right = None
        self.parent = None


class HierarchicalClustering(object):
    def __init__(self, train):
        self.train = train

    def calcNextTier(self, cluster_one, cluster_two):
        """
        :param cluster_one: [[], [], [], ...]
        :param cluster_two: [[], [], [], ...]
        :return:
        """
        distance = -sys.maxsize-1
        for idx1, node1 in enumerate(cluster_one):
            for idx2, node2 in enumerate(cluster_two):
                dis = sum([(node1[i] - node2[i])**2 for i in range(len(node1))])
                if dis > distance:
                    distance = dis
        return distance

    def herarchical(self, data):
        # initial : Each node is a cluster, traverse all the node
        crt = []
        for node in data:
            crt.append([node[0]]) # [ [[...]], [[...]], [[...]], ...]
        nxt = []

        tmp_cur_idx = 0
        tmp_nxt_idx = 0
        tier = 0
        print("Start tier: ", tier)
        print("Current tier node: ", crt)
        tier += 1
        while len(crt) != 1:
            distance = sys.maxsize
            print("Start tier :", tier)
            tier += 1
            # first cluster
            for cur_idx in range(len(crt)):

                # second cluster
                for nxt_idx in range(cur_idx+1, len(crt)):
                    # calc min distance of cluster
                    cluster2Distance = self.calcNextTier(crt[cur_idx], crt[nxt_idx])
                    if cluster2Distance < distance:
                        # Calc the distance between two cluster
                        distance = cluster2Distance
                        tmp_cur_idx = cur_idx
                        tmp_nxt_idx = nxt_idx
            print("Cluster :\n", crt[tmp_cur_idx], "\nAnd Cluster :\n", crt[tmp_nxt_idx], "\nMerge, the distance is :", distance)
            nxt.append(crt[tmp_cur_idx] + crt[tmp_nxt_idx])
            for i in range(len(crt)):
                if (i != tmp_cur_idx and i != tmp_nxt_idx):
                    nxt.append(crt[i])
            crt = nxt
            print("Current Tier Node: \n", crt, len(crt[0]))
            nxt = []

    def is_only_left_one(self, l, cur_idx):
        length = len(l)
        count = 0
        for i in range(len(l)):
            if l[i] is None and i != cur_idx:
                count += 1
        return (length - count) == 1

    def run(self):
        self.herarchical(self.train)


current = []

train = [[[170, 57, 32], 'W'],
         [[192, 95, 28], 'M'],
         [[150, 45, 30], 'W'],
         [[170, 65, 29], 'M'],
         [[175, 78, 35], 'M'],
         [[185, 90, 32], 'M'],
         [[170, 65, 28], 'W'],
         [[155, 48, 31], 'W'],
         [[160, 55, 30], 'W'],
         [[182, 80, 30], 'M'],
         [[175, 69, 28], 'W'],
         [[180, 80, 27], 'M'],
         [[160, 50, 31], 'W'],
         [[175, 72, 30], 'M']]
hc = HierarchicalClustering(train)
hc.run()
