# Linear Regression
import numpy as np
class linearRegression(object):
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
    
    def oneHotEncoding(self, y):
        # number of classes type
        classes = len(set(y))
        
        # format the target into 1-dimention
        targets = np.array(y).reshape(-1)
        
        # replace the numerated type by number
        for i in range(classes):
            targets[targets == list(set(targets))[i]] = i
            
        # change type to int
        targets = targets.astype(np.int64)
        
        # get one-hot matrix
        one_hot_targets = np.eye(classes)[targets]
        return one_hot_targets
    
    def lscombinesgd(self, X, y, step, max_iter_cnt):
        para = self.leastSquares(X, y)
        parameter = self.SGD(X, y, step, max_iter_cnt, para)
        return parameter
    
    def leastSquares(self, X, y): 
        # using the fact that \theta = (X^T X)^(-1)X^T y
        # set X_0 = 1
        one = np.ones(y.shape[0])
        X = np.c_[one, X]
        theta = np.ones(X.shape[1])
        parameter = np.dot(np.dot(np.matrix(np.dot(X.T, X)).I, X.T), y)
        return parameter
    
    
    def SGD(self, X, y, step, max_iter_cnt, theta = np.zeros((5, 3))):
        
        one = np.ones(y.shape[0])
        X = np.c_[one, X]
        m = X.shape[0]
        iter_cnt = 0
#         theta = np.zeros((5, 3))
#         theta = np.ones((5, 3))
        theta = np.array([[10, 3, 5],[8, 2, 5],[1, 5, 3],[2, 7, 5],[1, 2, 0]])
        iter_theta = []
        
        
        while (iter_cnt < max_iter_cnt):
            
            iter_theta.append(theta)
            
            # random select a data
            for rand in range(m):
                partialDerivitive = step*(np.dot(theta.T, X[rand].T) - y[rand].T).reshape((3, 1)) * X[rand].reshape((1, 5))
                theta = theta - partialDerivitive.T
            iter_cnt += 1
        return theta
    
    def BGD(self, X, y, step, max_iter_cnt):
        
        one = np.ones(y.shape[0])
        X = np.c_[one, X]
        m = X.shape[0]
        iter_cnt = 0
        theta = np.ones((5, 3))
        allSum = np.zeros((5, 3))
        iter_theta = []
                
        while (iter_cnt < max_iter_cnt):
            iter_theta.append(theta)
            for i in range(m):
                allSum += ((np.dot(theta.T, X[i].T) - y[i].T).reshape((3, 1)) * X[i].reshape((1, 5))).T
            theta = theta - step*allSum
            iter_cnt += 1
        return theta

    def k_fold(self, X, y, K):
        # Scramble the order of the data
        rand = np.random.RandomState(0)
        permutation = rand.permutation(X.shape[0])
        X, y = X[permutation], y[permutation]
        
        # Divide the data set into k parts.
#         each = 0
#         if ((X.shape[0]/K - X.shape[0]//K)>0):
#             each = int(X.shape[0]/K + 1)
#         else:
        each = int(X.shape[0]/K)
            
        accu = []
        # training K times
        for i in range(K):
            t = [j for j in range(y.shape[0])]
#             testIdx = np.array([j+i*each for j in range(each)])
            testIdx = np.array(t[0 + i*each: each + i*each])
            if (y.shape[0]-1 - testIdx[-1] < each):
                testIdx = np.array(t[i*each:])
            trainIdx = np.array([True]*X.shape[0])
            trainIdx[testIdx] = False
            train_X = X[trainIdx]
            train_y = y[trainIdx]
            test_X = X[testIdx]
            test_y = y[testIdx]

            """
            Three functions to get the parameter
            1. stochastic gradient descent
            2. the normal equations 
            3. combine the normal equations and stochastic gradient descent
            """
            parameter = self.SGD(train_X, train_y, 0.001, 1400)
#             parameter = self.leastSquares(train_X, train_y)
#             parameter = self.lscombinesgd(train_X, train_y, 0.001, 1400)

            one = np.ones(test_X.shape[0])
            test_X = np.c_[one, test_X]
            
            yx = np.dot(test_X, parameter)
            pre = np.argmax(yx,axis=1)
            one_hot_pre = np.eye(test_y.shape[1])[pre]
    
            p = 0
            for k in range(test_X.shape[0]):
#                 print(one_hot_pre[k] , test_y[k])
                if ((one_hot_pre[k] == test_y[k]).all()):
                    p += 1
            accu.append(p/test_y.shape[0])
            print(i, "-th part accurate :", p/test_y.shape[0])
        print("Divide into ", K, " parts.", "The accurate is : ", sum(accu)/K )
    
    
    def run(self):
        feature, label = self.load_csv("Iris.csv", ",")
        
        print("Using optimization funtion Stochastic Gradiant Descent:")
        for i in range(2, 11, 2):
            self.k_fold(feature, label, i)
    
linearRegression = linearRegression()
linearRegression.run()