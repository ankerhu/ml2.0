import matplotlib.pyplot as plt

decisionNode=dict(boxstyle="sawtooth",fc="0.8")
leafNode=dict(boxstyle="round4",fc="0.8")
arrow_args=dict(arrowstyle="<-")

def plotNode(nodeTxt,centerPt,parentPt,nodeType):
    createPlot.ax1.annotate(nodeTxt,xy=parentPt,xycoords='axes fraction',xytext=centerPt,textcoords='axes fraction',va='center',ha='center',bbox=nodeType,arrowprops=arrow_args)

def createPlot():
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    createPlot.ax1=plt.subplot(111,frameon=False)
    plotNode('a decision node',(0.5,0.1),(0.1,0.5),decisionNode)
    plotNode('a leaf node',(0.8,0.1),(0.3,0.8),leafNode)
    plt.show()

def getLeafsNum(myTree):
    leafsNum=0
    firtKey=list(myTree.keys())[0]
    secondDict=myTree[firtKey]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            leafsNum += getLeafsNum(secondDict[key])
        else:
            leafsNum += 1
    return leafsNum

def getTreeDepth(myTree):
    maxDepth=0
    firstKey=list(myTree.keys())[0]
    secondDict=myTree[firstKey]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            thisDepth=1+getTreeDepth(secondDict[key])
        else:
            thisDepth=1
    if thisDepth>maxDepth:
        maxDepth=thisDepth
    return maxDepth
def retrieveTree(i):
    listOfTrees =[{'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: 'no', 1: 'yes'}}}},
                  {'no surfacing': {0: 'no', 1: {'flippers': \
                    {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}}
                  ]
    return listOfTrees[i]
def plotMidText(centerPt,parentPt,txtString):
    midX=(parentPt[0]-centerPt[0])/2+centerPt[0]
    midY=(parentPt[1]-centerPt[1])/2+centerPt[0]
    createPlot.ax1.text(midX,midY,txtString)
def plotTree(myTree,parentPt,nodeTxt):
    leafsNum=getLeafsNum(myTree)
    treeDepth=getTreeDepth(myTree)
    firstStr=list(myTree.keys())[0]
    centerPt=(plotTree.xoff+(1+float(leafsNum))/2/plotTree.totalW,plotTree.yoff)
    plotMidText(centerPt,parentPt,nodeTxt)
    plotNode(firstStr,centerPt,parentPt,decisionNode)
    secondDict=myTree[firstStr]
    plotTree.yoff=plotTree.yoff-1.0/plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__=='dict':
            plotTree(secondDict[key],centerPt,str(key))
        else:
            plotTree.xoff=plotTree.xoff+1.0/plotTree.totalW
            plotNode(secondDict[key],(plotTree.xoff,plotTree.yoff),centerPt,leafNode)
            plotMidText((plotTree.xoff,plotTree.yoff),centerPt,str(key))
    plotTree.yoff=plotTree.yoff+1/plotTree.totalD
def createPlot(inTree):
    fig=plt.figure(1,facecolor='white')
    fig.clf()
    axprops=dict(xticks=[],yticks=[])
    createPlot.ax1=plt.subplot(111,frameon=False,**axprops)
    plotTree.totalW=float(getLeafsNum(inTree))
    plotTree.totalD=float(getTreeDepth(inTree))
    plotTree.xoff=-0.5/plotTree.totalW
    plotTree.yoff=1.0
    plotTree(inTree,(0.5,1.0),'')
    plt.show()