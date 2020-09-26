import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

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

print(numOfWordsA)
print(numOfWordsB)
