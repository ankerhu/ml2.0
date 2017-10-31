import bayes
postingList,classList=bayes.loadDataSet()
myVocabList=bayes.createVocabList(postingList)
print(postingList)
#myVocabVec=bayes.setOfWord2Vec(myVocabList,postingList[0])
#print(myVocabVec)
trainMat=[]
for postInDoc in postingList:
    trainMat.append(bayes.setOfWord2Vec(myVocabList,postInDoc))
#print(trainMat)
p0Vect,p1Vect,pAbusive=bayes.trainNB0(trainMat,classList)
print(p0Vect)
print(p1Vect)
print(pAbusive)