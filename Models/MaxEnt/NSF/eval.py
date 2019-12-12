

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn import metrics
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

	log_reg = pickle.load(open("maxentNSFModel.p","rb"))
	
	y_score = log_reg.decision_function(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	roc_auc = dict()

	for i in range(len(LABELS)):
		fpr[i], tpr[i], _ = metrics.roc_curve(Y_test , y_score[:, i],pos_label=i)
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	plt.figure()
	plt.plot([0, 1], [0, 1], 'k--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('NSF: MaxEnt Model Receiver operating characteristic curve')

	# Plot of a ROC curve for a specific class
	for i in range(len(LABELS)):
		plt.plot(fpr[i], tpr[i], label='ROC curve for label ' + str(i+1) + " " +list(LABELS.keys())[i]  +' (area = %0.2f)' % roc_auc[i])
		
		plt.legend(loc="lower right")
	
	plt.show()

	with open("maxentNSFModel.p","wb") as handle:

		pickle.dump(log_reg,handle)

	# Make prediction
	print("MAKING PREDICTIONS")
	Y_pred = log_reg.predict(X_test)

	with open("maxentNSFPredicted.p","wb") as handle:

		pickle.dump(Y_pred,handle)

	# print(Y_pred.tolist())

	# Calculate accuracy, precision, and recall
	print("PRINTING STATISTICS")
	acc = accuracy_score(y_true = Y_test, y_pred = Y_pred)
	prec = precision_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	recall = recall_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	print ("accuracy = " + str(acc))
	print ("precision = " + str(prec))
	print ("recall = " + str(recall))

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()