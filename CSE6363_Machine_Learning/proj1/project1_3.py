# GDA
import matplotlib.pyplot as plt
import numpy as np

class GDA(object):
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
    
    # calculate data's cov 
    def calcCov(self, X, y):
        m, n = X.shape
        data = np.c_[X, y.reshape(y.shape[0], 1)]
        class0_data = data[data[:, 3]==0]
        class1_data = data[data[:, 3]==1]
        u0 = np.mean(class0_data[:, :3], axis=0)
        u1 = np.mean(class1_data[:, :3], axis=0)
        class0_sub_u0 = class0_data[:, :3] - u0
        class1_sub_u1 = class1_data[:, :3] - u1
        
        x_sub_u = np.concatenate([class0_sub_u0,class1_sub_u1])
#         print (class0_sub_u0.shape, class1_sub_u1.shape, x_sub_u.shape) # (7, 3) (7, 3) (14, 3)
        x_sub_u=np.mat(x_sub_u)
        # cov
        sigma=(1.0/(m-2))*(x_sub_u.T*x_sub_u)
        phi=(1.0/m)*class1_data.shape[0]
#         print(phi, u0, u1, sigma)
        return phi, u0, u1, sigma, class0_data, class1_data
    
    def estimateProbability(self, X, phi, u0, u1, sigma):
        predict = []
        for i in X:
            logp0dividep1_1 = np.dot(    np.dot(np.linalg.inv(sigma), (u0-u1))    , i)
            logp0dividep1_2 = -(1/2)*np.dot(   np.dot((u0+u1), np.linalg.inv(sigma))    , (u0-u1))
        
            logp0dividep1 = logp0dividep1_1 + logp0dividep1_2
            # The probability of class 1
            if (logp0dividep1 > 0):
                print("0", "M")
            else:
                print("1", "W")
                
#         print(np.linalg.inv(sigma).shape, (u0-u1).shape, "*=======", -(1/2), np.dot(   np.dot((u0+u1), np.linalg.inv(sigma))    , (u0-u1)).shape)
        # print("y = ", np.dot(np.linalg.inv(sigma), (u0-u1)), "* x + ", -(1/2)*np.dot(   np.dot((u0+u1), np.linalg.inv(sigma))    , (u0-u1)))
        return np.dot(np.linalg.inv(sigma), (u0-u1)), -(1/2)*np.dot(   np.dot((u0+u1), np.linalg.inv(sigma))    , (u0-u1))
            
            # The probability of class 1
            
    def generateData(self, u0, u1, sigma):
        # use height and weight
        
        # generate two cluster data of Gaussian distributions
        # class 0
        u0 = u0[:2]
        cov = np.mat(sigma[:2, :2])
        x0 = np.random.multivariate_normal(u0,cov,50).T   #The first class point which labael equal 0
        y0 = np.zeros(np.shape(x0)[1])
        # class 1
        u1 = u1[:2]
        x1 = np.random.multivariate_normal(u1,cov,50).T   #The first class point which labael equal 0
        y1 = np.ones(np.shape(x1)[1])
        
        return x0, y0, x1, y1

    def show(self, x0, y0, x1, y1, class0_data, class1_data, phi, u0, u1, sigma, a, b):
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.set_title('Compare Generated data and Origin data')
        plt.xlabel('X')
        plt.ylabel('Y')
        ax1.scatter(x = x0[0], y = x0[1], color = 'red', marker = '+')
        ax1.scatter(x = class0_data[:, 0], y = class0_data[:, 1], color = 'blue', marker = '*')
        ax1.scatter(x = x1[0], y = x1[1], color = 'yellow', marker = '+')
        ax1.scatter(x = class1_data[:, 0], y = class1_data[:, 1], color = 'green', marker = '*')
        plt.legend('0o1i')
        
        
        plt.show()
         



    def run(self):
        train_X, train_y = self.split(self.train)
        test_X, test_y = self.split(self.test)
#         print(test_y) #["W":"1", "M":"0"]
        phi, u0, u1, sigma, class0_data, class1_data = self.calcCov(train_X, train_y)
        a, b = self.estimateProbability(test_X, phi, u0, u1, sigma)
        x0, y0, x1, y1 = self.generateData(u0, u1, sigma)
        self.show(x0, y0, x1, y1, class0_data, class1_data, phi, u0, u1, sigma, a, b)
        
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
    
test = [[[155, 40, 35], 'W'], 
        [[170, 70, 32], 'M'], 
        [[175, 70, 35], 'W'], 
        [[180, 90, 20], 'M']]

GDA = GDA(train, test)
GDA.run()



