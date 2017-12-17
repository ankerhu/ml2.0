import bayes
import feedparser
'''
postingList,classList=bayes.loadDataSet()
myVocabList=bayes.createVocabList(postingList)
print(myVocabList)
#myVocabVec=bayes.setOfWord2Vec(myVocabList,postingList[0])
#print(myVocabVec)
trainMat=[]
for postInDoc in postingList:
    trainMat.append(bayes.setOfWord2Vec(myVocabList,postInDoc))
#print(trainMat)
p0Vect,p1Vect,pAbusive=bayes.trainNB0(trainMat,classList)
#print(p0Vect)
#print(p1Vect)
#print(pAbusive)
testEntry=['love','my','dalmation']
thisDoc=bayes.setOfWord2Vec(myVocabList,testEntry)
print('{} classified as:{}'.format(testEntry,bayes.classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)))
testEntry=['stupid','garbage']
thisDoc=bayes.setOfWord2Vec(myVocabList,testEntry)
print('{} classified as:{}'.format(testEntry,bayes.classifyNB(thisDoc,p0Vect,p1Vect,pAbusive)))
'''

nasa=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
yahoo=feedparser.parse('http://www.nasa.gov/rss/dyn/image_of_the_day.rss')
#print(len(nasa['entries']))
#myVocabList,pNasa,pYahoo=bayes.localWords(nasa,yahoo)
bayes.getTopWords(nasa,yahoo)
#bayes.spamTest()
