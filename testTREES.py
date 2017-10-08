import trees
myDat,myLabels=trees.createDataSet()
myShannonEntropy=trees.calcShannonEntropy(myDat)
print(myShannonEntropy)
splitedDataSet=trees.splitDataSet(myDat,0,0)
print(splitedDataSet)
bestFeature=trees.chooseBestFeatureToSplit(myDat)
print(bestFeature)