import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
#stopwords to remove commonly used words
from nltk.corpus import stopwords

#Try and get dicts to exclude stopwords
stopwords.words('english')

#Term Frequency, num times a word appears in doc divided by total num of words in doc
def computeTF(wordDict, bagOfWords):
    tfDict = {}
    bagOfWordsCount = len(bagOfWords)
    for word, count in wordDict.items():
        tfDict[word] = count / float(bagOfWordsCount)
    return tfDict

#Inverse Data Frequency: log of num of docs divided by num of docs that contain certain word
def computeIDF(documents):
    import math
    N = len(documents)

    idfDict = dict.fromkeys(documents[0].keys(), 0)
    for document in documents:
        for word, val in document.items():
            if val > 0:
                idfDict[word] += 1
    for word, val in idfDict.items():
        idfDict[word] = math.log(N/float(val))
    return idfDict

documentA = 'the man went out for a walk'
documentB = 'the children sat around the fire'

bagOfWordsA = documentA.split(' ')
bagofWordsB = documentB.split(' ')

uniqueWords = set(bagOfWordsA).union(set(bagofWordsB))

numOfWordsA = dict.fromkeys(uniqueWords, 0)

for word in bagOfWordsA:
    numOfWordsA[word] += 1

numOfWordsB = dict.fromkeys(uniqueWords, 0)

for word in bagofWordsB:
    numOfWordsB[word] += 1

tfA = computeTF(numOfWordsA, bagOfWordsA)
tfB = computeTF(numOfWordsB, bagofWordsB)

#print(tfA)
#print(tfB)

idfs = computeIDF([numOfWordsA, numOfWordsB])
print(idfs)
