#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/2/26 10:10 AM
# @Author  : zhongch4g
# @Site    : 
# @File    : svd.py
# @Software: IntelliJ IDEA
import numpy as np
from numpy import linalg
import matplotlib.pyplot as plt
import heapq
import random

def rebuild_img(u, sigma, v, rank):
    # print("Rank: ", rank)
    # print(u.shape, sigma.shape, v.shape)
    m = len(u)
    n = len(v)
    re = np.zeros((m, n))

    # random select singular values
    # ranList = random.sample(range(rank+30), rank)

    # Select top-rank singular values 
    cur = 0
    while cur < rank:
        re[cur, cur] = sigma[cur]
        cur += 1
    
    # print("Select index of the singular values(in order): ", ranList)
    print("Select index of the singular values(in order): ", list(range(len(sigma[:rank]))))


    # reconstruct = np.dot(np.dot(u[:, ranList], np.diag(sigma[ranList])), v[ranList, :])
    reconstruct = np.dot(np.dot(u[:, :rank], re[:rank, :rank]), v[:rank, :])
    reconstruct[reconstruct < 0] = 0
    reconstruct[reconstruct > 255] = 255
    # set parameter type as uint8
    return reconstruct.astype("uint8"), (m*rank + rank*n + rank)
    # return reconstruct, (m*rank + rank*n + rank)


def svd(ruler, rank):
    """
    The eigenvalues(not ordered)
    The normalized (unit “length”) eigenvectors
    """
    m, n = ruler.shape
    # A*AT U
    b, U = linalg.eig(np.dot(ruler, ruler.T))
    # AT*A V
    sigma2, V = linalg.eig(np.dot(ruler.T, ruler))

    # sigma = np.sqrt(sigma2)

    # order the eigen value

    # U as col matrix V as row matrix
    # find the index of top K
    U_cur = []
    V_cur = []
    re = np.zeros((rank, rank))
    tempU = U
    tempS = sigma
    maxEigenValue = -99999
    maxEigenValueIdx = 0

    for i in range(rank):
        for j in range(tempU.shape[1]):
            if sigma[j] > maxEigenValue:
                maxEigenValue = sigma[j]
                maxEigenValueIdx = j
        # print("re[i, i]",re[i, i], "tempS[maxEigenValueIdx]", tempS[maxEigenValueIdx])
        re[i, i] = tempS[maxEigenValueIdx]
        U_cur.append(tempU[:, maxEigenValueIdx])
        tempS = np.delete(tempS, maxEigenValueIdx, axis=0)
        tempU = np.delete(tempU, maxEigenValueIdx, axis=1)
        maxEigenValue = -99999
    new_U = np.matrix(U_cur).T #(5, 512)

    tempV = V
    maxEigenValue = -99999
    maxEigenValueIdx = 0
    for i in range(rank):
        for j in range(tempV.shape[0]):
            if sigma[j] > maxEigenValue:
                maxEigenValue = sigma[j]
                maxEigenValueIdx = j
        V_cur.append(tempV[j, :])
        tempV = np.delete(tempV, j, axis=0)
        maxEigenValue = -99999
    new_V = np.matrix(V_cur) #(5, 512)
    # reconstruct = np.dot(np.dot(new_U, ), new_V)
    # reconstruct[reconstruct < 0] = 0
    # reconstruct[reconstruct > 255] = 255
    # # set parameter type as uint8
    # return reconstruct.astype("uint8")
    return 1
    
tif = plt.imread('./misc/ruler.512.tiff')
ruler = np.array(tif)
originpx = ruler.shape[0]*ruler.shape[1]
# origin image
plt.imshow(ruler, cmap='Greys_r')
plt.axis('off')
plt.show()
for rank in [5, 10, 20, 50, 100, 200]:
    u, sigma, v = np.linalg.svd(ruler)
    G, reformatePx = rebuild_img(u, sigma, v, rank)
    # G = svd(ruler, rank)
    print("Now rank is ===> ", rank)
    print("The space we saved is : ", originpx, "-", reformatePx, "=", originpx - reformatePx)
    print("Difference between origin and constructed:", np.sum(np.mat(ruler) - G))
    # set to grey
    plt.imshow(G, cmap='Greys_r')
    plt.axis('off')
    plt.show()


