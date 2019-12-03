import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from collections import OrderedDict
from multiprocessing import Pool

filename = "WOS.p"

# Top N entries to be found

topN = 500

# ngram length

n = 2

def countSentenceNGram(label, sentences, n):
	
	v_dict = {}

	for sentence in sentences:

		sentence = word_tokenize(sentence)
		sentence = ngrams(sentence, n)

		for i in sentence:
			wordsAdded = {}
			word = " ".join(i)
			if word not in v_dict:
				v_dict[word] = 0
			if word not in wordsAdded:
				wordsAdded[word] = 0
				v_dict[word] += 1

	sortedList = sorted(v_dict.keys() , key = lambda x: v_dict[x], reverse=True)

	return (label,v_dict,sortedList,n)

# Group abstracts into a dictionary where highest level label is the specialization and the lower is abstracts in that spec

def groupByLabelIntoDict(file):

	labels = OrderedDict()

	data = pickle.load(open(file, "rb"))

	for a in data:

		for l in a[1]:

			if l not in labels:

				labels[l] = []

			labels[l].append(a[0])

	return labels

ngramRes = 0 

def topNotIn(labelNgram):

	global topN

	global ngramRes

	counter = 0

	c = 0

	topNList = []

	while c < len(labelNgram[1]) and counter < topN:

		isNotIn = True

		for i2 in ngramRes:

			if(labelNgram[2][c][0] in i2[2][:c+1] and labelNgram[0] != i2[0]):

				isNotIn = False

		if(isNotIn): 

			topNList.append((labelNgram[2][c],labelNgram[1][labelNgram[2][c]]))

			counter += 1

		c += 1

	return labelNgram[0],topNList

def main(): 

	global filname

	global ngramRes

	global topN

	global n

	labels = groupByLabelIntoDict(filename)

	tuples = []

	sentences = []

	for key in list(labels.keys()):

		tuples.append((key,labels[key],n))

	ngramPool = Pool()

	map = ngramPool.starmap_async(countSentenceNGram,tuples)

	ngramPool.close()

	ngramPool.join()

	ngramRes = map.get(timeout=0)

	del labels

	del tuples

	print("Got ngram results")

	topNPool = Pool()
	
	map = topNPool.map_async(topNotIn,ngramRes)

	topNPool.close()

	topNPool.join()

	orderRes = map.get(timeout=0)

	for top in orderRes:

		print("----------- TOP",topN,"FOR",top[0],"-----------------")

		for i in range(len(top[1])):

			rank = i

			ngram = top[1][i][0]

			c_ngram = top[1][i][1]

			print(i,ngram,c_ngram)

		print("----------- TOP",topN,"FOR",top[0],"-----------------")

	exit()

if __name__ == '__main__':
	
	main()

exit()