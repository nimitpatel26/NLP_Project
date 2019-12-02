

import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams

filename = "WOS.p"
VOCAB = {}
LABELS = {}

# def countSentence(sentence, label):
# 	wordsAdded = []
# 	for i in sentence:
# 		key = VOCAB.get(label)
# 		if key == None:
# 			VOCAB[label] = {i:1}
# 			wordsAdded.append(i)
# 		else:
# 			keyVocab = key.get(i)
# 			if keyVocab == None:
# 				key[i] = 1
# 				wordsAdded.append(i)
# 			elif i not in wordsAdded:
# 				key[i] = keyVocab + 1
# 				wordsAdded.append(i)

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
		sentence = list(ngrams(sentence, 2))
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
			countSentenceNGram(sentence, j)

			# For unigram
			# countSentence(sentence, j)

	for i in VOCAB:
		sortedList = sorted(VOCAB[i].items(), key=lambda x: x[1], reverse=True)
		print("---------------------------")
		print(i + "\t" + str(LABELS[i]))
		print("---------------------------")
		for j in range(100):
			print(str(j + 1) + ". " + sortedList[j][0] + "\t" + str(sortedList[j][1]))
		print("")

main()
