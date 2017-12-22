from numpy  import *

def loadDataSet():
    dataMat=[]
    labelMat=[]
    with open('testSet.txt') as f:
       for line in f.readlines():
           lineArray=line.strip().split()
           dataMat.append([1.0,float(lineArray[0]),float(lineArray[1])])
           labelMat.append(lineArray[2])
    return dataMat,labelMat

def sigmoid(z):
    return 1.0/(1+exp(-z))

def gradAscent(dataMatIn,classLabels):
    dataMat = mat(dataMatIn)
    labelMat = mat(classLabels,dtype=int).transpose()
    m,n = shape(dataMat)
    weights = ones((n , 1))
    maxCircle=500
    magnitude = 0.001
    for i in range(maxCircle):
        #print(dataMat * weights)
        h=sigmoid(dataMat * weights)
        error = labelMat-h
        weights= weights + magnitude * dataMat.transpose() * error
    return weights

def stocGradAscent0(dataMatrix,classLabels):
    m,n = shape(dataMatrix)
    dataMatrix = array(dataMatrix,dtype=float)
    classLabels = array(classLabels,dtype=float)
    weights = ones(n)
    magnitude = 0.01
    for i in range(m):
        h = sigmoid(sum(dataMatrix[i] * weights))
        error = classLabels[i] - h
        weights = weights + magnitude *dataMatrix[i]*error
    return weights

def stocGradAscent1(dataMatrix,classLabel,numIter=150):
    m,n = shape(dataMatrix)
    dataMatrix = array(dataMatrix,dtype=float)
    classLabel = array(classLabel,dtype=float)
    weights = ones(n)
    for i in range(numIter):
        index = list(range(m))
        for j in range(m):
            magnitude = 4/(1.0+i+j) + 0.01
            stocIndex = int(random.uniform(0,len(index)))
            h = sigmoid(sum(dataMatrix[stocIndex]*weights))
            error = classLabel[stocIndex] - h
            weights = weights + magnitude * dataMatrix[stocIndex] * error
            del(index[stocIndex])
    return weights

def classifyVector(inX,weights):
    prob = sigmoid(sum(inX * weights))
    if  prob > 0.5:
        return 1.0
    else:
        return 0.0


def plotBestFit(wei):
    import matplotlib.pyplot as plt
    weights = wei #矩阵变二维数组
    dataMat , labelMat = loadDataSet()
    dataArray = array(dataMat)
    n = shape(dataArray)[0]
    xcord1 = []
    ycord1 = []
    xcord2 = []
    ycord2 = []
    for i in range(n):
        if int(labelMat[i]) == 1:
            xcord1.append(dataArray[i,1])
            ycord1.append(dataArray[i,2])
        else:
            xcord2.append(dataArray[i,1])
            ycord2.append(dataArray[i,2])
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(xcord1,ycord1,s=30,c='red',marker = 's')
    ax.scatter(xcord2,ycord2,s=30,c='green')
    x = arange(-3.0,3.0,0.1)
    y = (-weights[0]-weights[1]*x)/weights[2]
    ax.plot(x,y)
    plt.xlabel('X1')
    plt.ylabel('X2')
    plt.show()

def colicTest():
    frTrain = open('horseColicTraining.txt')
    frTest = open('horseColicTest.txt')
    trainingSet = []
    trainingLabels = []
    for line in frTrain.readlines():
        currentLine = line.strip().split('\t')
        lineArray = []
        for i in range(21):
            lineArray.append(float(currentLine[i]))
        trainingSet.append(lineArray)
        trainingLabels.append(float(currentLine[21]))
    trainWeights = stocGradAscent1(trainingSet,trainingLabels,500)
    errorCount = 0
    numTestVec = 0.0
    for line in frTest.readlines():
        numTestVec += 1.0
        currentLine = line.strip().split('\t')
        lineArray = []
        for i in range(21):
            lineArray.append(float(currentLine[i]))
        if int(classifyVector(array(lineArray),trainWeights)) != int(currentLine[21]):
            errorCount += 1
    errorRate = float(errorCount)/numTestVec
    print('The error rate of this test is : %f' % errorRate)
    return errorRate

def multiTest():
    numTests = 10
    errorSum = 0.0
    for k in range(numTests):
        errorSum += colicTest()
    print('after %d iterations the average error rate is : %f' % (numTests,errorSum/float(numTests)))