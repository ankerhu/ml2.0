from math import log
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