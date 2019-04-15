import numpy as np
import math
from matplotlib import pyplot as plt  
from mpl_toolkits.mplot3d import Axes3D

class lr(object):
    
    def load_csv(self, path, delimiter, test_path):
        # load data
        print("loading train set...\n")
        data = np.loadtxt(path, dtype=np.str, delimiter=delimiter)
        
        # set feature from str to float64
        feature = np.array(data[:, :-1].astype(np.float64))
        label = np.array(data[:, -1]).astype(np.float64)
        print("loading test set...\n")
        test_data = np.loadtxt(test_path, dtype=np.str, delimiter=delimiter)
        # set feature from str to float64
        test_feature = np.array(test_data[:, :-1].astype(np.float64))
        test_label = np.array(test_data[:, -1].astype(np.float64))
        return feature, label, test_feature, test_label
    
    def toOrder(self, X, order):
        orderX = []
        for i in X:
            tmp = []
            for j in range(order+1):
                for k in range(order+1):
                    if (j+k<=order):
                        tmp.append(math.pow(i[0], j)*math.pow(i[1], k))
            orderX.append(tmp)
#         print(np.array(orderX))
        return np.array(orderX)
    
    def leastSquares(self, X, y, order): 
            # using the fact that \theta = (X^T X)^(-1)X^T y
            # set X_0 = 1
            orderX = self.toOrder(X, order)
#             parameter = np.dot(np.dot(np.matrix(np.dot(orderX.T, orderX)).I, orderX.T), y)
#             print(orderX.shape, y.shape) #(40 36) 40
#             print("xxt-1",np.linalg.pinv(orderX).shape) # (36, 40)
            parameter = np.dot(np.linalg.pinv(orderX) , y.reshape((40,1)))
#             print(parameter) #(36, 1)
            return orderX, y, parameter
    
    def evaluate(self, test_feature, test_label, parameter, order):
        print("Reformat test feature to", order, "-th order...")
        
        orderX = self.toOrder(test_feature, order)
        predicty = np.dot(orderX,parameter) #(40, 36) (40, 36)

        tmp = test_label.reshape((test_feature.shape[0], 1))-predicty
        SSE = np.dot(tmp.T,tmp)
        print("SSE: ", SSE)        
            
    def show(self, feature, X, y, parameter, order):
        order = order
        fig = plt.figure()  
        ax = fig.add_subplot(111, projection='3d')
        # Generate regression surface
        x1 = np.arange(0, 10, 0.1)
        x2 = np.arange(0, 10, 0.1) 
        x1, x2 = np.meshgrid(x1, x2)
        z = np.ones((len(x1), len(x1)))
        for i in range(len(x1)):
            tmp = []
            for j in range(len(x2)):
                t = [[x1[i, j], x2[i, j]]]
                ox = self.toOrder(t, order)
                value = np.dot(ox, parameter)
                z[i, j] = value
                
        # plot the surface
        ax.plot_surface(x1, x2, z, rstride=1, cstride=1, cmap='rainbow') 
        # plot the dot that document given.
        ax.scatter(feature[:, 0], feature[:, 1], y, c='r')
        ax.set_zlabel('Z')
        ax.set_ylabel('Y')
        ax.set_xlabel('X')
        plt.show()
        
    def run(self):
        feature, label, test_feature, test_label = self.load_csv("project1.csv", ",", "project1_test.csv")
#         print(test_feature.shape)
        for i in [1, 2, 3, 4, 7]:
            print(i, "-th order result: ")
            orderX, y, parameter = self.leastSquares(feature, label, i)
            self.show(feature, orderX, y, parameter, i)
            self.evaluate(test_feature, test_label, parameter, i)
        
        
lr = lr()
lr.run()