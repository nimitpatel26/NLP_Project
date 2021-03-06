

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.naive_bayes import MultinomialNB
from scipy.sparse import dok_matrix,vstack,coo_matrix,csr_matrix
from collections import OrderedDict
from multiprocessing import Pool, Process
from nltk import word_tokenize
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
import multiprocessing as mp
import time
import numpy as np
import math

LABELS = OrderedDict({"CS":0, "Medical":1, "Civil":2, "ECE":3, "biochemistry":4, "MAE":5, "Psychology ":6})

def main():

	mainData = pickle.load(open("../../../Data/XY_WOS.p","rb"))

	X = mainData[0]

	Y = mainData[1]

	X_test = mainData[2]

	Y_test = mainData[3]

	del mainData

	nb = MultinomialNB()

	# Fit the model
	print("FITTING THE DATA")

	nb.fit(X,Y)

	with open("../../../Data/nbWOSModel.p","wb") as handle:

		pickle.dump(nb,handle)


# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()