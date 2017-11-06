from numpy import *
import sys

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
    listOfTokens=re.split(r'\W*',bigString)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def spamTest():
    docList=[]
    classList=[]
    fullText=[]
    #load and parse text files
    for i in range(1,26):
        wordList=textParse(open('email/spam/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
        wordList=textParse(open('email/ham/%d.txt' % i,errors='ignore').read())
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
    vocabList=createVocabList(docList)
    #randomly create the training set
    trainingSet=list(range(50))
    testSet=[]
    for i in range(10):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    #train the trainingSet
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(setOfWord2Vec(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0Vect,p1Vect,pSpam=trainNB0(trainMat,trainClasses)
    #classify the textSet and calculate the error rate
    errorCount=0
    errorTokenList=[]
    for docIndex in testSet:
        wordVector=setOfWord2Vec(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0Vect,p1Vect,pSpam) != classList[docIndex]:
            errorCount += 1
            errorTokenList.append(docList[docIndex])
    print('The error rate is :',float(errorCount)/len(testSet))
    if  errorCount != 0:
        print('Classification error ',errorTokenList)

def calcMostFreq(vocabList,fullText):
    '''
    caculates frequency of occurrence
    '''
    import operator
    freqDict={}
    for token in vocabList:
        freqDict[token]=fullText.count(token)
    sortedFreq=sorted(freqDict,key=operator.itemgetter(1),reverse=True)
    return sortedFreq[:30]

def localWords(feed0,feed1):
    import feedparser
    docList=[]
    classList=[]
    fullText=[]
    minLen=min(len(feed0['entries']),len(feed1['entries']))
    for i in range(minLen):
        wordList=textParse(feed0['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(0)
        wordList=textParse(feed1['entries'][i]['summary'])
        docList.append(wordList)
        fullText.extend(wordList)
        classList.append(1)
    vocabList=createVocabList(docList)
    top30Words=calcMostFreq(vocabList,fullText)
    for pairW in top30Words:
        if pairW[0] in vocabList:
            vocabList.remove(pairW[0])
    trainingSet=list(range(2*minLen))
    testSet=[]
    for i in range(20):
        randIndex=int(random.uniform(0,len(trainingSet)))
        testSet.append(trainingSet[randIndex])
        del(trainingSet[randIndex])
    trainMat=[]
    trainClasses=[]
    for docIndex in trainingSet:
        trainMat.append(bagOfWords2VecMN(vocabList,docList[docIndex]))
        trainClasses.append(classList[docIndex])
    p0Vect,p1Vect,pSpam=trainNB0(trainMat,trainClasses)
    errorCount=0
    for docIndex in testSet:
        wordVector=bagOfWords2VecMN(vocabList,docList[docIndex])
        if classifyNB(wordVector,p0Vect,p1Vect,pSpam) != classList[docIndex]:
            errorCount += 1
    print('the error rate is' , float(errorCount)/len(testSet))
    return vocabList,p0Vect,p1Vect