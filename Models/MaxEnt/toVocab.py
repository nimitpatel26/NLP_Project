

import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams

filename = "WOS.p"
VOCAB = {}
LABELS = {}

def countSentence(sentence, label):
	wordsAdded = []
	for i in sentence:
		key = VOCAB.get(label)
		if key == None:
			VOCAB[label] = {i:1}
			wordsAdded.append(i)
		else:
			keyVocab = key.get(i)
			if keyVocab == None:
				key[i] = 1
				wordsAdded.append(i)
			elif i not in wordsAdded:
				key[i] = keyVocab + 1
				wordsAdded.append(i)

def countSentenceNGram(sentence, label):
	wordsAdded = []
	for i in sentence:
		word = " ".join(list(i))
		key = VOCAB.get(label)
		if key == None:
			VOCAB[label] = {word:1}
			wordsAdded.append(word)
		else:
			keyVocab = key.get(word)
			if keyVocab == None:
				key[word] = 1
				wordsAdded.append(word)
			elif word not in wordsAdded:
				key[word] = keyVocab + 1
				wordsAdded.append(word)

def main():
	mainData = pickle.load(open(filename, "rb"))
	for i in mainData:
		sentence = list(word_tokenize(i[0]))

		# Comment for unigram
		# sentence = list(ngrams(sentence, 6))
		# print(list(sixgrams))
		# print("")
		labels = i[1]
		for j in labels:
			key = LABELS.get(j)
			if key == None:
				LABELS[j] = 1 
			else:
				LABELS[j] = key + 1

			# For n gram
			# countSentenceNGram(sentence, j)

			# For unigram
			countSentence(sentence, j)
	vocabLists = []
	vocabListsVocab = []
	labels = []
	for i in VOCAB:
		sortedList = list(sorted(VOCAB[i].items(), key=lambda x: x[1], reverse=True))
		vocabLists.append(sortedList)
		labels.append(i)
	for i in vocabLists:
		tmpList = []
		for j in i:
			tmpList.append(j[0])
		vocabListsVocab.append(tmpList)
	toPrint = {}
	# last = ""
	for i in range(len(vocabLists)):
		counter = 0
		j = 0
		# print("---------------------------")
		# print(labels[i] + "\t" + str(LABELS[labels[i]]))
		# print("---------------------------")
		while counter < 100:
			notInOthers = True
			for k in range(len(vocabLists)):
				if i != k and vocabLists[i][j][0] in vocabListsVocab[k][:j + 1]:
					notInOthers = False
				

			if notInOthers:
				# print(str(counter + 1) + ". " + vocabLists[i][j][0] + "\t" + str(vocabLists[i][j][1]))
				counter = counter + 1
				toPrint[vocabLists[i][j][0]] = 0
				# last = vocabLists[i][j][0]
				j = j + 1
			else:
				j = j + 1
		# print("")
	print ("VOCAB = {", end = "")
	for i in toPrint:
		print ("\"" + i + "\", ", end = "")
		# if i == last:
		# 	print (i, end = "")
		# else:
		# 	print (i + ", ", end = "")
	print("}")
			

main()
