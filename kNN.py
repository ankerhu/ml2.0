#! /usr/bin/python3
# coding=utf-8
from numpy import *
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
        #print(voteIlabel) 距离由短到长得到对应标签的值
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
        #print(classCount) 得到标签及其数量的字典列表
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    #print(classCount.items())items方法将字典变为元组列表
    #print(sortedClassCount) 将ClassCount按照第二个元素也就是标签出现的次数进行由大到小排序，越大说明在前k个标签中出现的重复次数越多
    return sortedClassCount[0][0]
def file2matrix(filename):
    fr=open(filename)
    numberOfLines=len(fr.readlines())
    returnMat=zeros((numberOfLines,3))
    classLabelVector=range(numberOfLines)
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