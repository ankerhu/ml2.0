import trees
import treePlotter
myDat,myLabels=trees.createDataSet()
'''

myShannonEntropy=trees.calcShannonEntropy(myDat)
print(myShannonEntropy)
splitedDataSet=trees.splitDataSet(myDat,0,0)
print(splitedDataSet)
bestFeature=trees.chooseBestFeatureToSplit(myDat)
print(bestFeature)
myTree=trees.createTree(myDat,myLabels)
print(myTree)
print(myLabels)

#treePlotter1.createPlot()
myTree['no surfacing'][3]='maybe'
print(myTree)
print(treePlotter.getLeafsNum(myTree))
print(treePlotter.getTreeDepth(myTree))
#treePlotter.createPlot(myTree)
#trees.classify(myTree,myLabels,[1,0])
print(myLabels)
'''
myTree=treePlotter.retrieveTree(0)

print(trees.classify(myTree,myLabels,[1,1]))
