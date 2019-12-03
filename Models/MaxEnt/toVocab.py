import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from collections import OrderedDict
from multiprocessing import Pool,Manager
import multiprocessing as mp
import gc

filename = "./arXivSpecMergedTokens.p"

# Top N entries to be found

topN = 500

# ngram length

n = 2

# if your data has already been tokenized and split into ngrams with n matching that above

pretokened = True

def countSentenceNGram(sentences):

	global n

	v_dict = {}

	count = 0

	for sentence in sentences[1]:

		if(not pretokened):

			sentence = list(word_tokenize(sentence))

		else:

			sentence = sentence.split(" ") 

		for i in range((len(sentence)-n)):
			wordsAdded = {}
			word = " ".join(sentence[i:i+n])
			# print(" ".join(sentence[i:i+n]))
			if word not in v_dict:
				v_dict[word] = 0
			if word not in wordsAdded:
				wordsAdded[word] = 0
				v_dict[word] += 1

		if(count % 10000 == 0):

			print(sentences[0],count,"/",len(sentences[1]))

		count += 1

	sortedList = sorted(v_dict.keys() , key = lambda x: v_dict[x], reverse=True)

	return (sentences[0],v_dict,sortedList,n)

# Group abstracts into a dictionary where highest level label is the specialization and the lower is abstracts in that spec

def groupByLabelIntoDict(f):

	labels = OrderedDict()

	data = pickle.load(open(f, "rb"))

	for a in data:

		for l in a[1]:

			if l not in labels:

				labels[l] = []

			labels[l].append(a[0])

	del data

	return labels

def topNotIn(i,ngramRes):

	global topN

	counter = 0

	c = 0

	topNList = []

	while c < len(ngramRes[i][2]) and counter < topN:

		ngram = ngramRes[i][2][c]

		c_ngram = ngramRes[i][1][ngram]

		isNotIn = True

		for i2 in ngramRes:

			if(ngram in i2[1] and ngramRes[i][0] != i2[0] and c_ngram <= i2[1][ngram]):

				isNotIn = False

		if(isNotIn): 

			topNList.append((ngramRes[i][2][c],ngramRes[i][1][ngramRes[i][2][c]]))

			counter += 1

			if(counter%(topN/10) == 0):

				print(ngramRes[i][0],counter)

		c += 1

	print("Completed")

	return ngramRes[i][0],topNList

if __name__ == '__main__':

	mp.set_start_method("spawn")

	# manager = Manager()

	groupedDict = groupByLabelIntoDict(filename)

	# tuples = manager.list()

	tuples = []

	for k in groupedDict:

		tuples.append((k,groupedDict[k]))

	# tuples = [(list(groupedDict.keys())[0],groupedDict[list(groupedDict.keys())[0]])]

	del groupedDict

	groupedDict = {}

	ngramPool = mp.Pool()

	gc.collect()

	map = ngramPool.map_async(countSentenceNGram,tuples)

	ngramPool.close()

	ngramPool.join()

	ngramRes = map.get(timeout=0)

	del tuples

	exit()

	print("Got ngram results")

	# ngramRes = manager.list(list(ngramRes))

	ngramRes = list(ngramRes)

	topNPool = mp.Pool()

	map = []
	
	for i in range(len(ngramRes)):

		map.append(topNPool.apply_async(topNotIn,args=(i,ngramRes)))

	topNPool.close()

	topNPool.join()

	orderRes = []

	for p in map:

		orderRes.append(p.get(timeout=0))

	del ngramRes

	for top in orderRes:

		print("----------- TOP",topN,"FOR",top[0],"-----------------")

		for i in range(len(top[1])):

			rank = i

			ngram = top[1][i][0]

			c_ngram = top[1][i][1]

			print(i,ngram,c_ngram)

		print("----------- TOP",topN,"FOR",top[0],"-----------------")

# exit()