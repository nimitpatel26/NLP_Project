

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.sparse import dok_matrix,vstack
from collections import OrderedDict
from multiprocessing import Pool, Process
import multiprocessing as mp
import time
import numpy as np
import math

VOCAB = list(pickle.load(open("top500arXivSpecMergedTokens1grams.p","rb"))) + list(pickle.load(open("top500arXivSpecMergedTokens2grams.p","rb")))
LABELS = {'math': 0, 'physics': 1, 'nlin': 2, 'q-bio': 3,
          'cs': 4, 'stat': 5, 'q-fin': 6, 'econ': 7, 'eess': 8}

# def countVocab(sentence, label):
# 	wordsAdded = []
# 	for i in sentence:
# 		key = DATA.get(label)
# 		if key == None:
# 			DATA[label] = {i:1}
# 			wordsAdded.append(i)
# 		else:
# 			keyVocab = key.get(i)
# 			if keyVocab == None:
# 				key[i] = 1
# 			elif i not in wordsAdded:
# 				key[i] = keyVocab + 1
# 				wordsAdded.append(i)

# def getData():
# 	None

featureDict = OrderedDict()

for j in range(0,len(VOCAB)):
	featureDict[VOCAB[j]] = j

def abstractFeatureLabel(abstract):

	global featureDict
	global LABELS

	X = dok_matrix((1,len(VOCAB)))
	Y = dok_matrix((1,1))

	for j in abstract[0]:

		if j in featureDict:

			X[0,featureDict[j]] = 1

	Y[0,0] = LABELS[abstract[1][0]]

	return X,Y

featureN = [1,2]

def abstractFeatureLabels(abstracts):

	global featureN
	global featureDict
	global LABELS

	X = dok_matrix((len(abstracts),len(VOCAB)))
	# Y =  [ [ 0 for i in range(0,1) ] for j in range(0,len(abstracts)) ]
	Y = []

	# a = some abstract number
	
	for a in range(0,len(abstracts)):

		sentence = abstracts[a][0].split(" ")

		# j = some token in the abstract

		for n in featureN:

			for i in range((len(sentence)-n+1)):

				gram = " ".join(sentence[i:i+n])

				if gram in featureDict:

					X[a,featureDict[gram]] = 1

				del gram

				gram = None

		del sentence

		sentence = None

		Y.append(LABELS[abstracts[a][1][0]])

	del abstracts

	abstracts = None

	return X,Y

# Instead of creating 20 data splits, create a split for each abstract in the data
def featureArraysNSplits(data):

	argtuples = []

	for a in data:
		argtuples.append((a[0],a[1]))

	abstractParsingPool = Pool(processes=10)
	
	map = abstractParsingPool.map_async(abstractFeatureLabel,argtuples)

	abstractParsingPool.close()

	abstractParsingPool.join()

	del argtuples
	
	argtuples = None
	
	res = map.get(timeout=0)

	return res


def main():

	mp.set_start_method("spawn")

	start = time.time()

	mainData = pickle.load(open("arXivSpecMergedTokens.p", "rb"))

	# split the data array into lists of tuples, each 1/20th the size of the original data

	argtuples20 = []

	for a in range(0,20):

		argtuples20.append(list([]))

	for a in range(0,len(mainData)):

		argtuples20[a%20].append((mainData[a][0],mainData[a][1]))

	del mainData

	mainData = None

	# each tuple list will get a process mapped to it, total of 20 processes in the pool

	abstractParsingPool = Pool(processes=10)

	map = abstractParsingPool.map_async(abstractFeatureLabels,argtuples20)

	abstractParsingPool.close()

	abstractParsingPool.join()

	# get the results from the processes that processed the data
	
	res = map.get(timeout=0)

	del argtuples20

	argtuples20 = None

	print("Got result in \t" +str(time.time()-start) + " s")

	# merge sparse lists for X and merge the label lists for Y using generators

	X = vstack([res[i][0] for i in range(0,len(res)) if i % 10 != 0])
	Y = [item for sublist in range(0,len(res)) for item in res[sublist][1] if sublist % 10 != 0]
	# Y = [res[sublist][1] for sublist in range(0,len(res)) if sublist % 10 != 0]

	print("Got training in \t" +str(time.time()-start) + " s")

	X_test = vstack([res[i][0] for i in range(0,len(res)) if i % 10 == 0])
	# Y_test = vstack([res[i][1] for i in range(0,len(res)) if i % 10 == 0],format="csr")
	Y_test = [item for sublist in range(0,len(res)) for item in res[sublist][1] if sublist % 10 == 0]

	print("Got test in \t" +str(time.time()-start) + " s")

	del mainData

	del argtuples20

	# print(X)

	# print(Y)

	log_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1, max_iter=1000,n_jobs=-1)

	# Fit the model
	print("FITTING THE DATA")

	log_reg.fit(X,Y)

	# Make prediction
	print("MAKING PREDICTIONS")
	Y_pred = log_reg.predict(X_test)

	# print(Y_pred.tolist())

	# Calculate accuracy, precision, and recall
	print("PRINTING STATISTICS")
	acc = accuracy_score(y_true = Y_test, y_pred = Y_pred)
	prec = precision_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	recall = recall_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	print ("accuracy = " + str(acc))
	print ("precision = " + str(prec))
	print ("recall = " + str(recall))

	# VOCAB = {}
	# for i in DATA:
	# 	print ("--------------------------")
	# 	print (i + "\t" + str(FILES_PER_LABEL[i]))
	# 	print ("--------------------------")

	# 	sortedList = sorted(DATA[i].items(), key=lambda x: x[1], reverse=True)
	# 	for j in range(500):
	# 		print(str(j + 1) + ".\t" + sortedList[j][0] + "\t\t" + str(sortedList[j][1]))
			# VOCAB[sortedList[j][0]] = 0
		# print("")
	# print("[", end = "")
	# for i in VOCAB:
	# 	print ("\"" + i + "\", ", end = "")
	# print("]")

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()
