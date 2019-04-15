#Gaussian Naive Bayes
import math
class GNB:
    def __init__(self, ipt, train):
        self.ipt = ipt
        self.train = train
    
    def gaussFunction(self, x, mu, squareOfSigma):
        p = (1/((2*3.14*squareOfSigma)**0.5))*math.exp(-(x-mu)**2/(2*squareOfSigma))
        return p
    
    def gaussianNaiveBayesClassifier(self, ipt, train):
        feature = [feat[0] for feat in self.train]
        label = [lbl[1] for lbl in self.train]
        setlabel = set(label)
        judgeDict = dict()
        for i in setlabel:
            judgeDict.setdefault(i, 1)
        
        labelp = dict()
        for i in label:
            labelp.setdefault(i, 0)
            labelp[i] += 1
            
        for i in setlabel:
            labelp[i] = labelp[i]/len(label)
        
        # calc mu and square of sigma
        GaussParaDict = self.calcGaussianPara(feature, label)
        
        # enum sex label
        for j in setlabel:
            # ipt is a single test set
            for i in range(len(ipt)):
#                 print("In Label ", j)
                p = self.gaussFunction(ipt[i], GaussParaDict[i][j][0], GaussParaDict[i][j][1])
#                 print("Number ", i, "-th Feature P = : ", p)    
                judgeDict[j] *= p
            judgeDict[j] *= labelp[j]
#         print("judgeDict =========> ", judgeDict)
#         print("It's Sex More Likely a :", max(judgeDict,key=judgeDict.get))
        result, tmp = 'I', -9999
        for i in judgeDict.keys():
            if judgeDict[i] > tmp:
                result = i
                tmp = judgeDict[i]
        return result
        
    def calcGaussianPara(self, feature, label):
        # count the number of different types of labels
        labeldict = dict()
        for i in label:
            labeldict.setdefault(i, 0)
            labeldict[i] += 1
#         print(labeldict)
        
        # calculate mu and square of sigma
        mu, squareOfSigma = 0, 0
        featureNum = len(feature[0])
        featuredict = dict()
        for i in range(featureNum):
            featuredict.setdefault(i, {})
        
        for l in labeldict.keys():
            ll = list(filter(lambda x: x[1] == l, zip(feature, label)))
            tmp = [0] * featureNum
            for i in ll:
                tmp = list(map(lambda x:x[0]+x[1], list(zip(tmp, i[0]))))
            # tmp: each col sum
#             print("tmp", tmp)
#             print("ll", ll)
            
            mu = [k/labeldict[l] for k in tmp]
#             print("mu = ", mu)
            
            tmp2 = [0] * featureNum
            tmp3 = []
            
            for i in ll:
                # tmp3: each col minus square of mu
#                 print("zip(i[0], mu))", list(zip(i[0], mu)))
                tmp3 = [(k[0] - k[1]) ** 2 for k in zip(i[0], mu)]
#                 print(tmp3)
                # tmp2: each col's sum after minus square of mu
                tmp2 = list(map(lambda x:x[0]+x[1], list(zip(tmp2, tmp3))))

            squareOfSigma = [k/(labeldict[l]-1) for k in tmp2]
#             print("squareOfSigma", squareOfSigma)
            
            for k in range(len(mu)):
                featuredict[k].setdefault(l, [])
                # make sure first one is mu and second one is square of sigma
                featuredict[k][l].append(mu[k])
                featuredict[k][l].append(squareOfSigma[k])
#         print("featuredict = ", featuredict)
        return featuredict
            
        
    def run(self):
        print("################## Test : ", i, "#######################")
        return self.gaussianNaiveBayesClassifier(i, self.train)


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

train1 = [[[170, 57], 'W'], 
         [[192, 95], 'M'], 
         [[150, 45], 'W'], 
         [[170, 65], 'M'], 
         [[175, 78], 'M'], 
         [[185, 90], 'M'], 
         [[170, 65], 'W'], 
         [[155, 48], 'W'], 
         [[160, 55], 'W'], 
         [[182, 80], 'M'], 
         [[175, 69], 'W'], 
         [[180, 80], 'M'], 
         [[160, 50], 'W'],
         [[175, 72], 'M']]

# test = [[155, 40, 35], [170, 70, 32], [175, 70, 35], [180, 90, 20]]
# test1 = [[155, 40], [170, 70], [175, 70], [180, 90]]
# gnb = GNB(test1, train1)
# gnb.run()

a = int(input("how many test case: "))
for j in range(a):
    i = [int(x) for x in raw_input("input the test case split by space(example:170 57 32): ").split()]
    gnb = GNB(i, train)
    predict = gnb.run()
    print(i, "It's Sex More Likely a :", predict)
