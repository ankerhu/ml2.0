#! /usr/bin/python3
# coding=utf-8
from numpy import *
from os import listdir
import operator
def createDataSet():
    group=array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    labels=['A','A','B','B']
    return group, labels
def classify0(inX,dataSet,labels,k):
    dataSetSize=dataSet.shape[0]
    diffMatt=tile(inX,(dataSetSize,1))-dataSet
    sqDiffMatt=diffMatt**2
    sqDistance=sqDiffMatt.sum(axis=1)
    distance=sqDistance**0.5
    #return distance 得到inX到每个点的距离
    sortedDistIndices=distance.argsort()
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndices[i]]
        #print(voteIlabel) #距离由短到长得到对应标签的值
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        #print(classCount) #得到标签及其数量的字典列表
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #print(classCount.items())#items方法将字典变为元组列表
    #print(sortedClassCount) #将ClassCount按照第二个元素也就是不同标签出现的次数进行由大到小排序，越大说明在前k个标签中出现的重复次数越多，如果是同一个标签就不存在排序。所以该算法的核心并不是找最近的标签，而且找在k个最近中出现概率最大的标签
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))
    classLabelVector=list(range(numberOfLines))
    fr=open(filename)
    index=0
    for line in fr.readlines():
        line=line.strip()
        listFromLine=line.split('\t')
        returnMat[index,:]=listFromLine[0:3]
        classLabelVector[index]=listFromLine[-1]
        index+=1
        #classLabelVector.append(listFromLine[-1])
    return returnMat,classLabelVector
def autoNorm(dataSet):
    minVals=dataSet.min(0)
    maxVals=dataSet.max(0)
    range=maxVals-minVals
    normDataSet=zeros(shape(dataSet))
    m=dataSet.shape[0]
    normDataSet=dataSet-tile(minVals,(m,1))
    normDataSet=normDataSet/tile(range,(m,1))
    return normDataSet,range,minVals
def datingClassSet():
    hoRatio=0.10
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    m=normMat.shape[0]
    errorCount=0.0
    numTestVect=int(m*hoRatio)
    for i in range(numTestVect):
        classifierResult=classify0(normMat[i,:],normMat[numTestVect:m,:],datingLabels[numTestVect:m],3)
        print('the classifier came back with :%s,the real answer is :%s'%(classifierResult,datingLabels[i]))
        if classifierResult!=datingLabels[i]:
            errorCount+=1.0
    print('the error rate is %f'%(errorCount/float(numTestVect)))
def classifyPerson():
    #resultList=['not at all','in a small doses','in a large doses']
    percentTats=float(input('percentage of time spent playing games'))
    ffMiles=float(input('frequent flier miles earned per year'))
    iceCream=float(input('liters of ice cream consumed per year'))
    inArr=array([percentTats,ffMiles,iceCream])
    datingDataMat,datingLabels=file2matrix('datingTestSet.txt')
    normMat,ranges,minVals=autoNorm(datingDataMat)
    #print(normMat)
    #print(ranges)
    #print(minVals)
    #print(datingLabels)
    classifierResult=classify0((inArr-minVals)/ranges,normMat,datingLabels,3)
    print('You will probably like the person ',classifierResult)
def img2vector(fileName):
    returnVect=zeros((1,1024))
    fr=open(fileName)
    for i in range(32):
        lineStr=fr.readline()
        for j in range(32):
            returnVect[0,32*i+j]=int(lineStr[j])
    return returnVect
def handwritingClassTest():
    hwLabel=[]
    trainingFileList=listdir('digits/trainingDigits')
    m=len(trainingFileList)
    trainingMat=zeros((m,1024))
    for i in range(m):
        fileNameStr=trainingFileList[i]
        fileName=fileNameStr.split('.')[0]
        classNumStr=int(fileName.split('_')[0])
        hwLabel.append(classNumStr)
        trainingMat[i,:]=img2vector('digits/trainingDigits/%s' % fileNameStr)
    testFileList=listdir('digits/testDigits')
    errorCount=0.0
    testM=len(testFileList)
    for i in range(testM):
        fileNameStr=testFileList[i]
        fileName=fileNameStr.split('.')[0]
        classNumStr=int(fileName.split('_')[0])
        testMat=img2vector('digits/testDigits/%s'%fileNameStr)
        classifierResult=classify0(testMat,trainingMat,hwLabel,3)
        print('the classifier came back with %d,the real answer is %d'% (classifierResult,classNumStr))
        if  classifierResult!=classNumStr:
            errorCount+=1.0
    errorRate=errorCount/float(testM)
    print('The errorRate is %s'%errorRate)

