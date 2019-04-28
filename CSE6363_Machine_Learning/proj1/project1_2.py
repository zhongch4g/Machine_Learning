import numpy as np


class logisticRegression(object):
    def __init__(self, train, test):
        self.train = train
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
    
    # use log likelihood 
    def BGD(self, X, y, step, itertimes):
        m, n = X.shape
        theta = np.ones((X.shape[1]+1, 1))
#         print("Init theta:", theta)
        x0 = np.ones((X.shape[0], 1))
        X = np.c_[X, x0]
        iter_cnt = 0
        while (iter_cnt < itertimes):

            # sum of gradiant
            allsum = np.zeros((4,))
            for i in range(X.shape[0]):
                allsum = allsum + (y[i]-self.sigmoid(np.dot(theta.T, X[i]))) * X[i]
            
            theta = theta + step * allsum.reshape((4, 1))

            iter_cnt += 1
            print(theta.reshape((1, 4)))
        
        return theta
    
    def predict(self, test_X, test_y, theta):
        x0 = np.ones((test_X.shape[0], 1))
        test_X = np.c_[test_X, x0]
        x_theta = np.dot(theta.T, test_X.T)
        pre = self.sigmoid(x_theta)
        print(pre)
        pre[pre>0.5] = 1
        pre[pre<=0.5] = 0
        return pre
    
    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))
    
    def run(self):
        train_X, train_y = self.split(self.train)
        test_X, test_y = self.split(self.test)
        theta = self.SGD(train_X, train_y, 0.01, 300)
        pre = self.predict(test_X, test_y, theta)
        for i in range(test_X.shape[0]):
            if pre[0, i] == 1:
                print(test_X[i], "W")
            else:
                print(test_X[i], "M")
    
# train = [[[170, 57, 32], 'W'],
#          [[192, 95, 28], 'M'],
#          [[150, 45, 30], 'W'],
#          [[170, 65, 29], 'M'],
#          [[175, 78, 35], 'M'],
#          [[185, 90, 32], 'M'],
#          [[170, 65, 28], 'W'],
#          [[155, 48, 31], 'W'],
#          [[160, 55, 30], 'W'],
#          [[182, 80, 30], 'M'],
#          [[175, 69, 28], 'W'],
#          [[180, 80, 27], 'M'],
#          [[160, 50, 31], 'W'],
#          [[175, 72, 30], 'M']]

train = [[[170, 57, 32], 'W'],
         [[192, 95, 28], 'M'],
         [[150, 45, 35], 'W'],
         [[168, 65, 29], 'M'],
         [[175, 78, 26], 'M'],
         [[185, 90, 32], 'M'],
         [[171, 65, 28], 'W'],
         [[155, 48, 31], 'W'],
         [[165, 60, 27], 'W']]
    
# test = [[[155, 40, 35], 'W'],
#         [[170, 70, 32], 'M'],
#         [[175, 70, 35], 'W'],
#         [[180, 90, 20], 'M']]

test = [[[169, 58, 30], 'W'],
        [[185, 90, 29], 'M'],
        [[148, 40, 31], 'W'],
        [[177, 80, 29], 'M'],
        [[170, 62, 27], 'W'],
        [[172, 72, 30], 'M'],
        [[175, 68, 27], 'W'],
        [[178, 80, 29], 'M']]


logisticRegression = logisticRegression(train, test)
logisticRegression.run()