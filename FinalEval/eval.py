

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
from sklearn.preprocessing import label_binarize
from sklearn import metrics
import multiprocessing as mp
import time
import numpy as np
import math
import os
from statistics import mean

def main():

    files = [f for f in os.listdir("../Data/ROC_Curves") if f.endswith(".p")]

    plt.figure()
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('All Models Average Receiver operating characteristic curve')

    for f in files:
        content = pickle.load(open("../Data/ROC_Curves/"+f,"rb"))
        fpr,tpr,_ = content[0]
        auc = content[1]
        plt.plot(fpr,tpr,label="ROC Curve for " +f.split(".")[0] +' (area = %0.2f)' % auc)
        plt.legend(loc="lower right")

    plt.show()

# prevent recursive multiprocessing in windows
if __name__ == '__main__':
	main()
