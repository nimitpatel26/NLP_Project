
from sklearn import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings

def main():
	cancer_data = datasets.load_breast_cancer()
	label_names = cancer_data['target_names']
	labels = cancer_data['target']
	feature_names = cancer_data['feature_names']
	features = cancer_data['data']

	cancer_df = pd.DataFrame(cancer_data.data, columns = cancer_data.feature_names)
	X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.30, random_state=42)

	### START CODING HERE ###
	# Import necessary module from sklearn.linear_model for Logistic Regression
	from sklearn.linear_model import LogisticRegression

	# Call LogisticRegression with a random_state of 0, solver='liblinear'
	logreg_clf = LogisticRegression(random_state=0, solver='liblinear')

	# Fit the model to X_train and Y_train
	logreg_clf = LogisticRegression.fit(logreg_clf, X=X_train, y=y_train)

	# Make predictions on X_test and store the results in y_pred
	test_predictionsLR = LogisticRegression.predict(logreg_clf, X=X_test)

	warnings.filterwarnings("ignore")

	### START CODING HERE ###
	# Import the necessary modules from sklearn for f1_score, roc_auc_score and cross_val_score
	from sklearn.metrics import f1_score
	from sklearn.metrics import roc_auc_score
	from sklearn.model_selection import cross_val_score

	# Compute the f1_score, roc_auc_score and cross_val_score and store them in properly named variables: 
	# f1_score_logreg, auc_logreg, cv_logreg
	# Hint: for cross validation with 5 folds, you should call the method on
	# (logreg_clf, cancer_data.data, cancer_data.target, cv=5)
	f1_score_logreg = f1_score(y_test, test_predictionsLR, average='binary')
	auc_logreg = roc_auc_score(y_test, test_predictionsLR)
	cv_logreg = cross_val_score(logreg_clf, cancer_data.data, cancer_data.target, cv=5)
	### END CODING HERE ###

	print("\nLogistic Regression")
	# Print the performance measures
	print("f1_score: ", f1_score_logreg)
	print("auc: ", auc_logreg)
	print("cross validation: ", cv_logreg.mean())


	# Import necessary module from sklearn for svm
	from sklearn import svm

	### START CODING HERE ###
	# Call svm.SVC with kernel='linear', gamma=10, C=10
	svm_clf = svm.SVC(kernel='linear', gamma=10, C=10)

	# Fit the model to X_train and Y_train
	svm_clf = svm.SVC.fit(svm_clf, X=X_train, y=y_train)

	# Make predictions on X_test and store the results in y_pred
	test_predictionsSVM = svm.SVC.predict(svm_clf, X=X_test)

	# Compute the f1_score, roc_auc_score and cross_val_score and store them in properly named variables: 
	# f1_score_svm, auc_svm, cv_svm
	# Hint: for cross validation with 5 folds, you should call the method on
	# (svm_clf, cancer_data.data, cancer_data.target, cv=5)

	f1_score_svm = f1_score(y_test, test_predictionsSVM, average='binary') 
	auc_svm = roc_auc_score(y_test, test_predictionsSVM)
	cv_svm = cross_val_score(svm_clf, cancer_data.data, cancer_data.target, cv=5)
	### END CODING HERE ###

	print("\nSVM")
	# Print the performance measures
	print("f1_score: ", f1_score_svm)
	print("auc: ", auc_svm)
	print("cross validation: ", cv_svm.mean())

	# Create two lists for C_values and f_scores
	C_values = range(1, 100)
	f_scores = []

	### START CODING HERE ###
	# Write a for loop that does the following steps:
	# iterate over c in C_values
	for c in C_values:
		# call svm model in each iteration passing a linear kernel, gamma=10 and the current c
		svm_clf = svm.SVC(kernel='linear', gamma=10, C=c)
		# fit the model on X_train and y-train
		svm_clf = svm.SVC.fit(svm_clf, X=X_train, y=y_train)
		# predict X_test and store them in y_pred
		test_predictionsSVM = svm.SVC.predict(svm_clf, X=X_test)
		# compute f1_score and append it to f_scores
		f1_score_svm = f1_score(y_test, test_predictionsSVM, average='binary')
		f_scores.append(f1_score_svm)
	### END CODING HERE ###

	plt.title('Impact of SVM C Parameter on f1_score')
	plt.xlabel('C', fontsize=14)
	plt.ylabel('f1_score', fontsize=14)
	plt.plot(C_values, f_scores)
	plt.show()

main()


