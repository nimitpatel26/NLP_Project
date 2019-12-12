

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.sparse import dok_matrix,vstack
from collections import OrderedDict
from multiprocessing import Pool, Process
from sklearn import metrics
import multiprocessing as mp
import time
import numpy as np
import math

LABELS = OrderedDict({'math': 0, 'physics': 1, 'nlin': 2, 'q-bio': 3,
          'cs': 4, 'stat': 5, 'q-fin': 6, 'econ': 7, 'eess': 8})

def main():

	mainData = pickle.load(open("XY_ARXIV.p","rb"))

	X = mainData[0]

	Y = mainData[1]

	X_test = mainData[2]

	Y_test = mainData[3]

	del mainData

	log_reg = LogisticRegression(multi_class="ovr",solver="lbfgs", C=.1, max_iter=10000,n_jobs=-1)

	# Fit the model
	print("FITTING THE DATA")

	log_reg.fit(X,Y)

	with open("maxentARXIVModel.p","wb") as handle:

		pickle.dump(log_reg,handle)

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()
