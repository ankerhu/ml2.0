from math import log
import operator
def majorityCount(classList):
    classCount={}
    for vote in classCount:
        if  vote not in classCount.keys():
            classCount[vote]=0
        classCount[vote]+=1
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
def calcShannonEntropy(dataSet):
    numEntries=len(dataSet)
    labelCount={}
    for featVec in dataSet:
        currentLabel=featVec[-1]
        if currentLabel not in labelCount.keys():
            labelCount[currentLabel]=0
        labelCount[currentLabel]+=1
    shannonEntropy=0.0
    for key in labelCount.keys():
        prob=float(labelCount[key])/numEntries
        shannonEntropy-=prob*log(prob,2)
    return shannonEntropy
def createDataSet():
    dataSet=[[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    labels = ['no surfacing','flippers']
    return dataSet,labels
def splitDataSet(dataSet,axis,value):
    returnDataSet=[]
    for featVec in dataSet:
        if  featVec[axis]==value:
            reducedFeatVec=featVec[:axis]
            reducedFeatVec.extend(featVec[axis+1:])
            returnDataSet.append(reducedFeatVec)
    return returnDataSet
def chooseBestFeatureToSplit(dataSet):
    featuresNum=len(dataSet[0])-1
    baseEntropy=calcShannonEntropy(dataSet)
    bestInfoGain=0.0
    bestFeature=-1
    for i in range(featuresNum):
        featureList=[example[i] for example in dataSet]
        uniqueVals=set(featureList)
        newEntropy=0.0
        for value in uniqueVals:
            subDataSet=splitDataSet(dataSet,i,value)
            prob=len(subDataSet)/float(len(dataSet))
            newEntropy+=prob*calcShannonEntropy(subDataSet)
        infoGain=baseEntropy-newEntropy
        if  bestInfoGain<infoGain:
            bestInfoGain=infoGain
            bestFeature=i
    return bestFeature
def createTree(dataSet,labels):
    classList=[example[-1] for example in dataSet]
    #firsts stopping condition
    if  classList.count(classList[0])==len(classList):
        return classList[0]
    #second stopping condition
    if len(dataSet[0])==1:
        return majorityCount(classList)
    #building a tree
    bestFeature=chooseBestFeatureToSplit(dataSet)
    bestFeatureLabel=labels[bestFeature]
    del(labels[bestFeature])
    myTree={bestFeatureLabel:{}}
    featureValue=[example[bestFeature] for example in dataSet]
    uniqueVals=set(featureValue)
    for value in uniqueVals:
        subLabels=labels[:]
        myTree[bestFeatureLabel][value]=createTree(splitDataSet(dataSet,bestFeature,value),subLabels)
    return myTree
def classify(inputTree,featLabels,testVec):
    firstStr=list(inputTree.keys())[0]
    secondDict=inputTree[firstStr]
    featIndex=featLabels.index(firstStr)
    for key in secondDict.keys():
        if testVec[featIndex]==key:
            if type(secondDict[key]).__name__=='dict':
                classlabel=classify(secondDict[key],featLabels,testVec)
            else:
                classlabel=secondDict[key]
    return classlabel