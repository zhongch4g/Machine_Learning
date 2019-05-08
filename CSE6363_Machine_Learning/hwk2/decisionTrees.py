import csv
import numpy as np


# Need Modify
def calcEntropy(y):
    Entropy = 0
    numOfdata = y.shape[0]
    eachClass = dict()
    for i in y:
        eachClass.setdefault(i, 0)
        eachClass[i] += 1
    for each in eachClass.values():
        Entropy += -(each/numOfdata)*np.log(each/numOfdata)
    print(Entropy)
    return Entropy
    
def calcConditionEntropy(X, y):
    Entropy = calcEntropy(y)
#     ConditionEntropy = 0
    numOfdata = y.shape[0]
    eachClass = dict()
    maxInfo = 0
    maxInfoFeature = 0
    # each feature
    for i in range(X.shape[1]):
        print(i, "-th feature")
        eachClass = dict()
        ConditionEntropy = 0
        # each feature each class's count
        # eachClass {feature class: {data class1: num1, data class2: num2 ...}}
        for idx, j in enumerate(X[:, i]):
            eachClass.setdefault(j, {})
            eachClass[j].setdefault(y[idx], 0)
            eachClass[j][y[idx]] += 1
        
        for each in eachClass.keys():
            count = 0
            for classcnt in eachClass[each].keys():
                count += eachClass[each][classcnt]
            eachClass[each]["sum"] = count   
            
        # each feature class
        for p in eachClass.keys():
            ratio = eachClass[p]["sum"]/numOfdata
            for q in eachClass[p].keys():
                # feature class / numOfdata
                if q != "sum":
                    ConditionEntropy += -ratio*(eachClass[p][q]/eachClass[p]["sum"])*np.log(eachClass[p][q]/eachClass[p]["sum"])
        print("ConditionEntropy: ", ConditionEntropy)
        InformationGain = Entropy - ConditionEntropy
#         print("Information Gain:" , InformationGain)
        if (InformationGain > maxInfo):
            maxInfo = InformationGain
            maxInfoFeature = i
    return maxInfoFeature

def splitDataSet(X, bestFeatureIdx, cls):
    slt = [i for i in range(X.shape[1])]
    del(slt[bestFeatureIdx])
    idx = np.where(X[:, bestFeatureIdx]==cls)

    subsetX = X[:, slt]
    subsetX = subsetX[idx[0], :]
    subsety = y[idx[0]]
    return subsetX, subsety
    

def generateDecitionTree(X, y, labelList):
    if (len(set(y)) == 1):
        return y[0]
    if (X.shape[1] == 1):
        unique, counts = np.unique(y, return_counts=True)
        tmp = dict(zip(unique, counts))
        label, cnt = -1, -1
        for key, value in tmp.items():
            if (value > cnt):
                cnt = value
                label = key
        return label
    bestFeatureIdx = calcConditionEntropy(X, y)
    bestFeature = labelList[bestFeatureIdx]
    # store in dict
    Tree = {bestFeature:{}}
    labelList = np.delete(labelList, bestFeatureIdx)
    featureData = X[:, bestFeatureIdx]
    for cls in set(featureData):
        subLabels = labelList
        nX, ny = splitDataSet(X, bestFeatureIdx, cls)
        Tree[bestFeature][cls]=generateDecitionTree(nX, ny, subLabels)
    return Tree

def classify(X, y, Tree, labelList):
    cnt = 0
    for i in range(X.shape[0]):
        testX = X[i]
        testy = y[i]
        predict = -1
        tmpTree = Tree
        while type(tmpTree) == dict:
            feat = list(tmpTree.keys())[0]
            feat_idx = labelList.index(feat)
            result = tmpTree[feat][testX[feat_idx]]
            if (type(result) == dict):
                tmpTree = result
            else:
                predict = result
                tmpTree = -1
        if (predict == testy):
            cnt += 1
    print("Accuracy: ", cnt/X.shape[0])

with open('MushroomTrain.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = np.array(list(reader))
    y = data[:, 0]
    X = data[:, 1:]
labelList = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor']

# use all the feature
Tree = generateDecitionTree(X, y, labelList)

# delete feature 'odor'
# X = X[:, :-1]
# labelList = labelList[:-1]
# Tree = generateDecitionTree(X, y, labelList)

print("classify in train set: ")
classify(X, y, Tree, labelList)

with open('MushroomTest.csv', 'r') as f:
    reader = csv.reader(f, delimiter=',')
    data = np.array(list(reader))
    y = data[:, 0]
    X = data[:, 1:]
labelList = ['cap-shape', 'cap-surface', 'cap-color', 'bruises', 'odor']

print("classify in test set: ")
classify(X, y, Tree, labelList)
