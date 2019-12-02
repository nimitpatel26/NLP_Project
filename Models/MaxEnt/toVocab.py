

import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
# from nltk import ngrams

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

def main():
	mainData = pickle.load(open(filename, "rb"))
	for i in mainData:
		sentence = word_tokenize(i[0])
		labels = i[1]
		for j in labels:
			# key = VOCAB.get(j)
			# if key == None:
			# 	VOCAB[j] = {}
			countSentence(sentence, j)
			# print(j)

		# print(i[1])

		# for j in sixgrams:
		# 	print (j)

	for i in VOCAB:
		sortedList = sorted(VOCAB[i].items(), key=lambda x: x[1], reverse=True)
		print("---------------------------")
		print(i)
		print("---------------------------")
		for j in range(100):
			print(str(j + 1) + ". " + sortedList[j][0] + "\t" + str(sortedList[j][1]))
		print("")

main()
