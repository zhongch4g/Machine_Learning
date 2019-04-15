#K-NN
import math
class KNN:
    def __init__(self, ipt, train, k):
        self.ipt = ipt
        self.train = train
        self.k = k
    def knn_classify(self, ipt, train, k):
        cnt = 0
        feature = [feat[0] for feat in self.train]
        label = [lbl[1] for lbl in self.train]
        distance = self.distance(ipt, feature)
        con = zip(distance, label)
        distance = sorted(con, key = lambda x:x[0])
        topk = distance[:k]
        for i in topk:
            if i[1] == "W":
                cnt += 1
        if cnt/k > 0.5:
            return "W"
        return "M"
    
    def distance(self, ipt, feature):
        diff = []
        for feat in feature:
            diff.append(sum([pow((ipt[i]-feat[i]), 2) for i in range(len(ipt))]))
        diff = list(map(lambda x:math.sqrt(x), diff))
        return diff
            
    
    def run(self):
        classes = self.knn_classify(self.ipt, self.train, self.k)
        return classes
        
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

# test = [[155, 40, 35], [170, 70, 32], [175, 70, 35], [180, 90, 20]]
# for i in test:
#     t = KNN(i, train, 1)
#     classes = t.run()
#     print(i, classes)
    
a = int(input("How many test case: "))
k = int(input("Please set your K value(1~20): "))
for j in range(a):
    i = [int(x) for x in raw_input("input the test case split by space(example:170 57 32): ").split()]
    t = KNN(i, train, k)
    classes = t.run()
    print("k = %d ,input = %s, classes = %s"%(k, i, classes) )
