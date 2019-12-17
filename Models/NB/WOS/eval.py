

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from scipy.sparse import dok_matrix,vstack,coo_matrix,csr_matrix
from collections import OrderedDict
from multiprocessing import Pool, Process
from nltk import word_tokenize
from sklearn import metrics
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import label_binarize
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

	nb = pickle.load(open("../../../Data/nbWOSModel.p","rb"))

	# Make prediction
	print("MAKING PREDICTIONS")

	Y_pred = nb.predict(X_test)

	y_score = nb.predict_proba(X_test)

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

	plt.title('WOS: Naive Bayes Model Receiver operating characteristic curve')

	# Plot of a ROC curve for a specific class
	for i in range(len(LABELS)):
		plt.plot(fpr[i], tpr[i], label='ROC curve for label ' + str(i+1) + " " +list(LABELS.keys())[i]  +' (area = %0.2f)' % roc_auc[i])
		
	plt.legend(loc="lower right")

	plt.show()

	with open("../../../Data/nbWOSPredicted.p","wb") as handle:

		pickle.dump(Y_pred,handle)

	with open("../../../Data/ROC_Curves/NB WOS.p","wb") as handle:
		curve = metrics.roc_curve(label_binarize(Y_test,classes=list(LABELS.values())).ravel(),y_score.ravel())
		auc = metrics.roc_auc_score(label_binarize(Y_test,classes=list(LABELS.values())),label_binarize(Y_pred,classes=list(LABELS.values())),average="micro")		
		pickle.dump((curve,auc),handle)

	# print(Y_pred.tolist())

	# Calculate accuracy, precision, and recall
	print("PRINTING STATISTICS")
	acc = accuracy_score(y_true = Y_test, y_pred = Y_pred)
	print ("accuracy = " + str(acc))
	print("Macro Averging")
	prec = precision_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	recall = recall_score(y_true = Y_test, y_pred = Y_pred, average = "macro")
	print("F1 score = " +str(metrics.f1_score(Y_test,Y_pred,average="macro")))
	print ("precision = " + str(prec))
	print ("recall = " + str(recall))
	print("Micro Averging")
	prec = precision_score(y_true = Y_test, y_pred = Y_pred, average = "micro")
	recall = recall_score(y_true = Y_test, y_pred = Y_pred, average = "micro")
	print("F1 score = " +str(metrics.f1_score(Y_test,Y_pred,average="micro")))
	print ("precision = " + str(prec))
	print ("recall = " + str(recall))

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()