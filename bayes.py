from numpy import *
def loadDataSet():
    postingList=[['my', 'dog', 'has', 'flea', \
                          'problems', 'help', 'please'],
                         ['maybe', 'not', 'take', 'him', \
                          'to', 'dog', 'park', 'stupid'],
                         ['my', 'dalmation', 'is', 'so', 'cute', \
                           'I', 'love', 'him'],
                         ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                         ['mr', 'licks', 'ate', 'my', 'steak', 'how',\
                           'to', 'stop', 'him'],
                         ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    classVec=[0,1,0,1,0,1]
    return postingList,classVec

def createVocabList(dataSet):
    vocabSet=set([])
    for document in dataSet:
        vocabSet=vocabSet | set(document)
    return list(vocabSet)

def setOfWord2Vec(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)]=1
        else:
            print('the word %d is not in my vocabulary!' % word)
    return returnVec

def bagOfWords2VecMN(vocabList,inputSet):
    returnVec=[0]*len(vocabList)
    for word in inputSet:
        if word in vocabList:
            returnVec[vocabList.index(word)] += 1
    return returnVec
    
def trainNB0(trainMatrix,trainCategory):
    numTrainDocs=len(trainMatrix)
    numWords=len(trainMatrix[0])
    pAbusive=sum(trainCategory)/numTrainDocs
    p0Num=ones(numWords)
    p1Num=ones(numWords)
    p0Denom=2.0
    p1Denom=2.0
    for i in range(numTrainDocs):
        if trainCategory[i] == 1:
            p1Num += trainMatrix[i]
            p1Denom += sum(trainMatrix[i])
        else:
            p0Num += trainMatrix[i]
            p0Denom += sum(trainMatrix[i])
    p1Vect=log(p1Num/p1Denom)
    p0Vect=log(p0Num/p0Denom)
    return p0Vect,p1Vect,pAbusive

def classifyNB(vec2Classify,p0Vect,p1Vect,pClass1):
    p1=sum(vec2Classify * p1Vect) + log(pClass1)
    p0=sum(vec2Classify * p0Vect) + log(1.0 - pClass1)
    if p1>p0:
        return 1
    else:
        return 0

def textParse(bigString):
    import re
    listOfTokens=re.split(r'\w*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt'%i).read())
        