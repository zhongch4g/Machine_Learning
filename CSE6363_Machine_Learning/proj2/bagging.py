import numpy as np
import random
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

                # print(theta.shape, X[rand].shape)

                theta = theta + step * (py - h) * X[rand].reshape((n+1, 1))
            iter_cnt += 1
        return theta
    
    def predict(self, test_X, theta):
        x0 = np.ones((test_X.shape[0], 1))
        test_X = np.c_[test_X, x0]
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta)
        pre[pre>0.5] = 1
        pre[pre<=0.5] = 0
        return pre
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def bagging(self):
        train_X, train_y = self.load_data(self.train_path)
        test_X, test_y = self.load_data(self.test_path)

        theta = self.SGD(train_X, train_y, 0.1, 110)
        pre = self.predict(test_X, theta)
        print("err rate: ", 1 - np.sum((pre==test_y)*1)/pre.shape[1])        
        print("=======================")

        pred = np.array([])
        # bagging 
        for bag in [10, 50, 100]:
            for size in range(bag):
                # Random sampling
                sample = np.random.choice(np.array([i for i in range(train_X.shape[0])]), size=train_X.shape[0], replace=True)

                train_s_X, train_s_y = train_X[sample, :], train_y[sample]
                # print(train_s_X, train_s_y)
                # print(train_X.shape, train_y.shape)
                # 0.01, 200 95%
                step = random.random()*0.1
                epoch = random.randint(100, 1500)
                print("step: ", step, "epoch: ", epoch)
                theta = self.SGD(train_s_X, train_s_y, step, epoch)
                # print(theta)
                pre = self.predict(test_X, theta)
                if (pred.size <= 0):
                    pred = np.array(pre)
                else:
                    pred = np.r_[pred, pre]
            # print(pred.shape, test_X.shape)
            cnt = 0
            for i in range(test_X.shape[0]):
                each = pred[:, i]
                # pred to 1
                if (sum(each)>(each.size/2)) and (test_y[i] == 1):
                    cnt += 1
                if (sum(each)<=(each.size/2)) and (test_y[i] == 0):
                    cnt += 1
            err_rate = 1 - cnt/test_X.shape[0]
            print("err rate: ", err_rate)




train_path = "./train.txt"
test_path = "./test.txt"
logisticRegression = logisticRegression(train_path, test_path)
logisticRegression.bagging()