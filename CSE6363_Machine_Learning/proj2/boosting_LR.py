#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2019/4/4 3:21 PM
# @Author  : zhongch4g
# @Site    : 
# @File    : boosting_LR.py
# @Software: IntelliJ IDEA

import numpy as np

class logisticRegression(object):
    def __init__(self, train_path, test_path):
        self.train_path = train_path
        self.test_path = test_path

    def load_data(self, data_path):
        feature = []
        label = []
        with open (data_path, "r") as r:
            allset = r.read().split("\n")
            for line in allset:
                data = line.split(",")
                feature.append(data[:-1])
                label.append(data[-1])
        return np.array(feature).astype("float64"), np.array(label).astype("float64")

    # use log likelihood
    def SGD(self, X, y, step, itertimes):
        m, n = X.shape
        theta = np.ones((X.shape[1], 1))
        iter_cnt = 0
        while (iter_cnt < itertimes):
            # Random select data
            for rand in range(X.shape[0]):
                py = (y[rand]/2) + 0.5
                h = self.sigmoid(np.dot(theta.T, X[rand]))
                theta = theta + step * (py - h) * X[rand].reshape((n, 1))
            iter_cnt += 1
        return theta

    def predict(self, test_X, theta):
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta)
        pre[pre>0.5] = 1
        pre[pre<=0.5] = -1
        return pre

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))

    def boosting(self, boosting_times = [10, 25, 50]):
        # In boosting, the data label will be 1 & -1
        train_X, train_y = self.load_data(self.train_path)
        x0 = np.ones((train_X.shape[0], 1))
        train_X = np.c_[train_X, x0]
        train_y[train_y == 0] = -1

        test_X, test_y = self.load_data(self.test_path)
        x1 = np.ones((test_X.shape[0], 1))
        test_X = np.c_[test_X, x1]
        test_y[test_y == 0] = -1

        theta = self.SGD(train_X, train_y, 0.1, 110)
        # print("LR theta: ", theta)
        pred = self.predict(test_X, theta)
        print("Single LR err rate: ", 1 - np.sum((pred == test_y)*1)/pred.shape[1])
        # print(times, "times boosting...")
        # Initial the weight of the data

        for times in boosting_times:
            model_coef = []
            model = []
            cls_err_rate = 0
            coef = np.array([1/train_X.shape[0]]*train_X.shape[0])
            diag_coef = np.diag(coef)
            weighted_data = np.dot(diag_coef, train_X)
            print("start ", times, "times boosting...")
            for i in range(times):
                theta = self.SGD(weighted_data, train_y, 0.1, 100)
                # print("Weighted theta: ", theta)
                model.append(theta.flatten().tolist())
                # predict in train_X
                pre = self.predict(train_X, theta)
                print(i, "-th clasifier err rate in train set: ", 1 - np.sum((pre == train_y)*1)/pre.shape[1])
            # classifier error rate
                compare = ((pre != train_y) * 1)[0, :]
                # print("compare: ", compare)
                cls_err_rate = np.dot(compare.T, coef)
                # print(i, "time, cls_err_rate: ", cls_err_rate)
                # print("cls_err_rate = ", cls_err_rate)
                alpha = (1 / 2)*np.log((1 - cls_err_rate) / cls_err_rate)
                # print("model coeficient: ", alpha)
                model_coef.append(alpha)
                # update weight of the data
                yG = (train_y * pre).flatten()
                Z = np.dot(coef, np.exp(-alpha*yG))
                # print("Z = ", Z)
                coef = ((coef / Z) * np.exp(-alpha*yG)).flatten()
                diag_coef = np.diag(coef)
                weighted_data = np.dot(diag_coef, train_X)
                if (cls_err_rate >= 0.5):
                    print("cls_err_rate approximitely to 0...")
                    break
            self.boosting_predict(test_X, test_y, model_coef, model, times)

    def boosting_predict(self, X, y, model_coef, model, times):
        pre_list = []
        for i in model:
            tmp = self.predict(X, np.array(i))
            pre_list.append(tmp)
        # print(pre_list)
        model_coef_diag = np.diag(model_coef)
        weighted_result = np.dot(model_coef_diag, np.array(pre_list))
        fres = [0]*X.shape[0]
        for result in weighted_result:
            fres += result
        finres = np.sign(fres)
        print(times, "times boosting err rate in test set: ", 1 - np.sum((finres==y)*1)/X.shape[0])


train_path = "./train.txt"
test_path = "./test.txt"
logisticRegression = logisticRegression(train_path, test_path)
times = [10, 25, 50]    
logisticRegression.boosting(times)


