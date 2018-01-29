import svmMLiA
from numpy import *
dataSet,labelSet=svmMLiA.loadData('testSet.txt')
print(mat(labelSet).T)
