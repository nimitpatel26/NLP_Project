

import pickle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from sklearn.preprocessing import label_binarize
import sklearn.metrics as metrics
import pandas as pd
from collections import OrderedDict
import tensorflow as tf
import nltk

DATA = {}
FILES_PER_LABEL = {}
# VOCAB = {"this", "In", "we", "can", "data", "We", "it", "method", "such", "algorithm", "our", "new", "show", "two", "approach", "computer", "information", "present", "one", "methods", "applications", "problem", "propose", "To", "number", "algorithms", "software", "each", "provide", "processing", "computing", "many", "Our", "set", "large", "learning", "techniques", "image", "some", "several", "security", "existing", "various", "efficient", "while", "problems", "solutions", "features", "demonstrate", "programming", "parallel", "distributed", "framework", "where", "implementation", "structures", "provides", "accuracy", "improve", "multiple", "complex", "then", "computational", "better", "key", "very", "experiments", "computation", "approaches", "any", "them", "same", "real", "images", "available", "networks", "detection", "As", "terms", "result", "possible", "need", "tool", "knowledge", "so", "user", "allows", "evaluation", "way", "traditional", "space", "With", "functions", "Moreover", "databases", "users", "called", "vision", "tools", "database", "and", ")", "(", "were", "was", "study", ":", "patients", "%", "Results", ";", "associated", "may", "treatment", "Methods", "disease", "significant", "significantly", "=", "clinical", "had", "years", "compared", "increased", "risk", "after", "age", "all", "Conclusions", "higher", "Background", "2", "health", "most", "1", "<", "no", "factors", "group", "total", "included", "reported", "Conclusion", "groups", "3", "lower", "performed", "therapy", "care", "respectively", "symptoms", "common", "patient", "assessed", "population", "should", "without", "women", "p", "conducted", "aim", "95", "2017", "There", "outcomes", "association", "diagnosis", "changes", "evaluated", "mean", "diseases", "chronic", "P", "evaluate", "Objective", "primary", "assess", "determine", "[", "]", "10", "regression", "months", "investigate", "decreased", "measured", "4", "reduced", "CI", "healthy", "identify", "cases", "period", "6", "5", "type", "treated", "prevalence", "aimed", "Patients", "All", "water", "different", "use", "construction", "through", "management", "development", "under", "potential", "due", "building", "material", "area", "environmental", "three", "application", "environment", "over", "main", "field", "technology", "harvesting", "increase", "case", "quality", "B.V.", "areas", "pollution", "effective", "green", "smart", "rainwater", "project", "change", "solar", "soil", "bridge", "monitoring", "future", "geotextile", "analyzed", "2015", "sensing", "projects", "increasing", "sustainable", "natural", "up", "buildings", "tests", "remote", "resources", "structural", "local", "value", "assessment", "indicate", "suspension", "made", "technologies", "capacity", "industry", "Based", "demand", "spatial", "purpose", "2014", "activities", "urban", "economic", "average", "China", "region", "global", "ambient", "scale", "sources", "climate", "land", "impacts", "aims", "Water", "storage", "rainfall", "-", "objective", "concentration", "improvement", "layer", "sensors", "alternative", "sustainability", "2013", "benefits", "mainly", "promising", "bridges", "composite", "series", "is", "are", "paper", "This", "system", "proposed", "using", "control", "power", "A", "model", "performance", "systems", "electrical", "presented", "time", "presents", "current", "energy", "parameters", "order", "simulation", "signal", "circuit", "voltage", "frequency", "controller", "output", "designed", "network", "low", "circuits", "applied", "technique", "dynamic", "input", "electric", "digital", "linear", "range", "operation", "response", "implemented", "state", "load", "nonlinear", "motor", "operational", "conventional", "amplifier", "analog", "solution", "shown", "consumption", "filter", "cost", "electricity", "PID", "simulations", "identification", "optimization", "generation", "stability", "supply", "An", "Finally", "representation", "speed", "optimal", "simple", "effectiveness", "CMOS", "integrated", "given", "generator", "active", "error", "gain", "microcontroller", "components", "operating", "required", "controlled", "scheme", "achieve", "measurement", "phase", "reduce", "converter", "parameter", "feedback", "suitable", "signals", "noise", "derived", "dynamics", "proposes", "devices", "mu", "mode", "from", "have", "has", "these", "been", "also", "expression", "cells", "cell", "analysis", "gene", "human", "protein", "its", "blotting", "molecular", "showed", "role", "found", "These", "important", "biology", "However", "reaction", "including", "levels", "genes", "chain", "identified", "activity", "could", "metabolism", "revealed", "genetic", "review", "novel", "DNA", "polymerase", "first", "Here", "proteins", "observed", "function", "specific", "detected", "species", "mechanisms", "involved", "PCR", "further", "addition", "samples", "level", "major", "expressed", "demonstrated", "acid", "known", "cancer", "tissue", "understanding", "similar", "Northern", "Southern", "growth", "presence", "confirmed", "enzyme", "vitro", "mRNA", "mechanism", "highly", "factor", "biological", "tissues", "immune", "assay", "quantitative", "determined", "RNA", "functional", "single", "recent", "small", "Furthermore", "sequence", "target", "cellular", "enzymes", "indicated", "regulation", "pathways", "vivo", "blood", "characterized", "induced", "pathway", "via", "therapeutic", "transcription", "on", "which", "at", "C", "design", "Elsevier", "rights", "reserved", "materials", "into", "process", "high", "2016", "engineering", "developed", "well", "conditions", "flow", "work", "Ltd.", "It", "experimental", "models", "obtained", "properties", "machine", "temperature", "surface", "efficiency", "manufacturing", "out", "fluid", "engine", "strength", "structure", "investigated", "pressure", "internal", "combustion", "will", "within", "characteristics", "mechanical", "considered", "numerical", "thermal", "heat", "rate", "processes", "test", "For", "studied", "carried", "Published", "mechanics", "ratio", "stress", "modeling", "production", "measurements", "size", "density", "maximum", "Engineering", "distribution", "gas", "equations", "fuel", "theory", "element", "finite", "because", "good", "computer-aided", "velocity", "reduction", "calculated", "point", "hydraulics", "volume", "critical", "along", "degrees", "methodology", "machines", "part", "thermodynamics", "hydraulic", "3D", "form", "means", "University", "geometry", "mass", "basis", "boundary", "equation", "Ltd", "account", "cycle", ",", "that", "or", "between", "not", "their", "more", "research", "but", "'s", "than", "both", "studies", "'", "social", "findings", "other", "effects", "about", "during", "participants", "who", "when", "disorder", "they", "how", "effect", "whether", "suggest", "behavior", "among", "related", "examined", "only", "evidence", "differences", "those", "relationship", "disorders", "individuals", "positive", "support", "sample", "attention", "there", "less", "negative", "children", "people", "cognitive", "gender", "did", "impact", "across", "influence", "measures", "literature", "examine", "discussed", "emotional", "depression", "article", "greater", "personality", "mental", "early", "analyses", "individual", "implications", "life", "context", "likely", "''", "behaviors", "completed", "``", "often", "task", "being", "physical", "Participants", "communication", "brain", "child", "psychological", "Although", "general", "roles", "others", "either", "students", "behavioral", "female", "relationships", "exposure", "memory", "experience", "adults", "suggests", "however", }
VOCAB = list(pickle.load(open("../../../Data/top500relabeledNSFfiltered1grams.p","rb"))) + list(pickle.load(open("../../../Data/top500relabeledNSFfiltered2grams.p","rb")))

NUM_LABELS = 9
NUM_FEATURES = len(VOCAB)
BATCH_SIZE = 500
EPOCHS = 20
STEPS_PER_EPOCH = 100

LABELS = {'ENG': 0, 'MCS': 1, 'EAOS': 2, 'PS': 3, 'BAS': 4, 'SOC': 5, 'PSY': 6, 'ET': 7, 'HLT': 8}

# creates batches from sparse X data(scipy csr_matrix format) and dense y data to hopefully save on memory
# if batch size is reasonably small, this should save memory quite a bit I think
def nn_batch_generator(X_data, y_data, batch_size):
	
	samples_per_epoch = X_data.shape[0]
	number_of_batches = samples_per_epoch/batch_size
	counter=0
	
	index = np.arange(y_data.shape[0])

	while 1:
		index_batch = index[batch_size*counter:batch_size*(counter+1)]
		X_batch = X_data[index_batch,:].todense()
		# y_batch = [[0 if x != y else 1 for x in range(7)] for y in y_data[index_batch]]
		# keras.utils.np_utils.to.categorical()
		y_batch = y_data[index_batch]
		y_batch = tf.keras.utils.to_categorical(y_batch,num_classes=NUM_LABELS)
		counter += 1
		yield np.array(X_batch),y_batch
		if (counter > number_of_batches):
			counter=0

def countVocab(sentence, label):
	wordsAdded = []
	for i in sentence:
		key = DATA.get(label)
		if key == None:
			DATA[label] = {i:1}
			wordsAdded.append(i)
		else:
			keyVocab = key.get(i)
			if keyVocab == None:
				key[i] = 1
			elif i not in wordsAdded:
				key[i] = keyVocab + 1
				wordsAdded.append(i)

def getData():
	None

def main():

    mainData = pickle.load(open("../../../Data/XY_NSF.p", "rb"))

    X = mainData[0]
    Y = mainData[1]
    X_test = mainData[2]
    Y_test = mainData[3]

    nn_clf = tf.keras.models.load_model("../../../Data/dnnNSFModel.h5")
    nn_clf_history = pickle.load(open("../../../Data/dnnNSFTrainHistory.p","rb"))
    df = pd.DataFrame(nn_clf_history).plot(figsize=(10,5),title="NSF: DNN Model Training History")
    df.set_xlabel("Training Epoch")

    # Make prediction
    print("MAKING PREDICTIONS")
    # Y_pred = log_reg.predict(X_test)
    classPredict = nn_clf.predict_classes(X_test)
    y_score = nn_clf.predict(X_test)

    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    nn_threshold = dict()
    roc_auc = dict()

    for i in range(len(LABELS)):
        fpr[i], tpr[i], _  = metrics.roc_curve(np.array(Y_test) , y_score[:,i] , pos_label=i)
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])

    plt.figure()
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('NSF: DNN Model Receiver operating characteristic curve')

    # Plot of a ROC curve for a specific class
    for i in range(len(LABELS)):
        plt.plot(fpr[i], tpr[i], label='ROC curve for label ' + str(i+1) + " " +list(LABELS.keys())[i]  +' (area = %0.2f)' % roc_auc[i])
        plt.legend(loc="lower right")

    plt.show()

    with open("../../../Data/ROC_Curves/DNN NSF.p","wb") as handle:		
        pickle.dump((metrics.roc_curve(label_binarize(Y_test,classes=list(LABELS.values())).ravel(),y_score.ravel()),metrics.roc_auc_score(label_binarize(Y_test,classes=list(LABELS.values())),label_binarize(classPredict,classes=list(LABELS.values())))),handle)

    # Calculate accuracy, precision, and recall
    print("PRINTING STATISTICS")
    acc = accuracy_score(y_true = Y_test, y_pred = classPredict)
    prec = precision_score(y_true = Y_test, y_pred = classPredict, average = "macro")
    recall = recall_score(y_true = Y_test, y_pred = classPredict, average = "macro")
    print ("accuracy = " + str(acc))
    print ("precision = " + str(prec))
    print ("recall = " + str(recall))
	
main()
