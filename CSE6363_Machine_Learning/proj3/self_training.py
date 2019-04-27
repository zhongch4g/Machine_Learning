#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/25 3:10 PM
# @Author  : zhongch4g
# @Site    : 
# @File    : self_training.py
# @Software: IntelliJ IDEA
import numpy as np


class SelfTraining(object):
    def __init__(self, train, unlabeled, test):
        self.train = train
        self.unlabeled = unlabeled
        self.test = test

    # factorize
    def tointclass(self, label):
        intclass = np.array(np.unique(label, return_inverse=True)[1]).astype(np.float64)
        return intclass

    def split(self, data):
        feature = [feat[0] for feat in data]
        label = self.tointclass([lbl[1] for lbl in data])
        return np.array(feature), np.array(label)

    # use log likelihood
    def SGD(self, X, y, step, itertimes):
        m, n = X.shape
        theta = np.ones((X.shape[1]+1, 1))
        #         print("Init theta:", theta)
        x0 = np.ones((X.shape[0], 1))
        X = np.c_[X, x0]
        iter_cnt = 0
        while (iter_cnt < itertimes):
            # Random select data
            for rand in range(X.shape[0]):
                py = y[rand]
                h = self.sigmoid(np.dot(theta.T, X[rand]))
                #                 print(theta.shape, X[rand].shape)
                theta = theta + step * (py - h) * X[rand].reshape((4, 1))
            iter_cnt += 1
        return theta

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def predict(self, train_X, train_y, test_X, theta, k):
        # print(theta.T.shape, test_X.T.shape)
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta).flatten().tolist()
        test_X = test_X.tolist()
        train_X = train_X.tolist()
        train_y = train_y.tolist()
        new_test = [[x[:-1], y] for (x,y) in sorted(zip(test_X, pre), key=lambda pair:pair[1], reverse=True)]

        for i in range(k):
            if (len(new_test) == 0):
                break
            if (new_test[0][1] > 0.5):
                train_X.append(new_test[0][0])
                train_y.append(1)
                new_test.pop(0)
            else:
                train_X.append(new_test[0][0])
                train_y.append(0)
                new_test.pop(0)
        new_test = [i[0] for i in new_test]
        return np.array(train_X), np.array(train_y), new_test

    def predict_LR(self, test_X, test_y, theta):
        x0 = np.ones((test_X.shape[0], 1))
        test_X = np.c_[test_X, x0]
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta)
        pre[pre>0.5] = 1
        pre[pre<=0.5] = 0
        return pre

    def predict_semi(self, test_X, test_y, theta):
        x0 = np.ones((test_X.shape[0], 1))
        test_X = np.c_[test_X, x0]
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta)
        pre[pre>0.5] = 1
        pre[pre<=0.5] = 0
        return pre

    def run(self, epoch):
        train_X, train_y = self.split(self.train)
        test_X, test_y = self.split(self.test)

        x0 = np.ones((np.array(self.unlabeled).shape[0], 1))
        unlabel = np.c_[self.unlabeled, x0]

        length = train_X.shape[0] + unlabel.shape[0]

        theta = 0
        # Semi supervised model
        while(len(unlabel) != 0 or train_X.shape[0] != length):
            theta = self.SGD(train_X, train_y, 0.01, epoch)
            train_X, train_y, unlabel = self.predict(train_X, train_y, unlabel, theta, k = 2)
            x0 = np.ones((np.array(unlabel).shape[0], 1))
            unlabel = np.c_[unlabel, x0]

        pre = self.predict_semi(test_X, test_y, theta)
        correct = 0
        for i in range(test_X.shape[0]):
            if pre[0, i] == 1 and test_y[i] == 1:
                correct += 1
            if pre[0, i] == 0 and test_y[i] == 0:
                correct += 1
            if pre[0, i] == 1:
                print(test_X[i], "W")
            if pre[0, i] == 0:
                print(test_X[i], "M")
        accuracy = correct / test_X.shape[0]
        print("The accuracy of semi-supervised classifier: ", accuracy)


        # LR model
        train_X, train_y = self.split(self.train)
        theta = self.SGD(train_X, train_y, 0.01, epoch)
        pre = self.predict_LR(test_X, test_y, theta)
        correct = 0
        for i in range(test_X.shape[0]):
            if pre[0, i] == 1 and test_y[i] == 1:
                correct += 1
            if pre[0, i] == 0 and test_y[i] == 0:
                correct += 1
            if pre[0, i] == 1:
                print(test_X[i], "W")
            if pre[0, i] == 0:
                print(test_X[i], "M")
        accuracy = correct / test_X.shape[0]
        print("The accuracy of single logic regression classifier: ", accuracy)





train = [[[170, 57, 32], 'W'],
         [[192, 95, 28], 'M'],
         [[150, 45, 35], 'W'],
         [[168, 65, 29], 'M'],
         [[175, 78, 26], 'M'],
         [[185, 90, 32], 'M'],
         [[171, 65, 28], 'W'],
         [[155, 48, 31], 'W'],
         [[165, 60, 27], 'W']]


unlabeled = [[182, 80, 30],
        [175, 69, 28],
        [178, 80, 27],
        [160, 50, 31],
        [170, 72, 30],
        [152, 45, 29],
        [177, 79, 28],
        [171, 62, 27],
        [185, 90, 30],
        [181, 83, 28],
        [168, 59, 24],
        [158, 45, 28],
        [178, 82, 28],
        [165, 55, 30],
        [162, 58, 28],
        [180, 80, 29],
        [173, 75, 28],
        [172, 65, 27],
        [160, 51, 29],
        [178, 77, 28],
        [182, 84, 27],
        [175, 67, 28],
        [163, 50, 27],
        [177, 80, 30],
        [170, 65, 28]]

test = [[[169, 58, 30], 'W'],
        [[185, 90, 29], 'M'],
        [[148, 40, 31], 'W'],
        [[177, 80, 29], 'M'],
        [[170, 62, 27], 'W'],
        [[172, 72, 30], 'M'],
        [[175, 68, 27], 'W'],
        [[178, 80, 29], 'M']]

st = SelfTraining(train, unlabeled, test)

for i in [200, 500]:
    print("Epoch: ", i)
    st.run(i)