

import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from collections import OrderedDict
from multiprocessing import Pool

filename = "WOS.p"
VOCAB = {}
LABELS = OrderedDict()

def countSentenceNGram(label, sentences, n):
	
	VOCAB = {}

	VOCAB[label] = {}

	for sentence in sentences:

		sentence = list(word_tokenize(sentence))
		sentence = set(list(ngrams(sentence, n)))

		wordsAdded = {}
		for i in sentence:
			word = " ".join(list(i))
			if word not in VOCAB[label]:
				VOCAB[label][word] = 0
			VOCAB[label][word] += 1

	sortedList = sorted(VOCAB[label].items(), key=lambda x: x[1], reverse=True)

	return (label,sortedList,n)

def topLevelDict():

	mainData = pickle.load(open(filename, "rb"))

	for a in mainData:

		for l in a[1]:

			if l not in LABELS:

				LABELS[l] = []

			LABELS[l].append(a[0])
	
topLevelDict()

keys = []

sentences = []

n = 2

for key in list(LABELS.keys()):

	keys.append((key,LABELS[key],2))

ngramPool = Pool()

map = ngramPool.starmap_async(countSentenceNGram,keys)

ngramPool.close()

ngramPool.join()

res = map.get(timeout=0)


print(res[0][0])
print(res[0][1][:100])


exit()