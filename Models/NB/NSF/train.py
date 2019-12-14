

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.sparse import dok_matrix,vstack,coo_matrix,csr_matrix
from collections import OrderedDict
from multiprocessing import Pool, Process
from nltk import word_tokenize
import multiprocessing as mp
import time
import numpy as np
import math
import matplotlib.pyplot as plt

LABELS = OrderedDict({'ENG': 0, 'MCS': 1, 'EAOS': 2, 'PS': 3, 'BAS': 4, 'SOC': 5, 'PSY': 6, 'ET': 7, 'HLT': 8})

def main():

	mainData = pickle.load(open("XY_NSF.p","rb"))

	X = mainData[0]

	Y = mainData[1]

	X_test = mainData[2]

	Y_test = mainData[3]

	del mainData

	# print(X)

	# print(Y)

	nb = MultinomialNB()

	# Fit the model
	print("FITTING THE DATA")

	fitRes = nb.fit(X,Y)

	with open("nbNSFModel.p","wb") as handle:

		pickle.dump(nb,handle)

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()