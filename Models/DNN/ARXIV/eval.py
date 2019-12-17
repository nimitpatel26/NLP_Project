

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
VOCAB = list(pickle.load(open("../../../Data/top500arXivSpecMergedTokens1grams.p","rb"))) + list(pickle.load(open("../../../Data/top500arXivSpecMergedTokens2grams.p","rb")))
LABELS = {'math': 0, 'physics': 1, 'nlin': 2, 'q-bio': 3,
          'cs': 4, 'stat': 5, 'q-fin': 6, 'econ': 7, 'eess': 8}

NUM_LABELS = len(list(LABELS.keys()))
NUM_FEATURES = len(VOCAB)
BATCH_SIZE = 500
EPOCHS = 20
STEPS_PER_EPOCH = 100

def main():

	mainData = pickle.load(open("../../../Data/XY_ARXIV.p", "rb"))

	X = mainData[0]
	Y = mainData[1]
	X_test = mainData[2]
	Y_test = mainData[3]

	nn_clf = tf.keras.models.load_model("../../../Data/dnnARXIVModel.h5")
	nn_clf_history = pickle.load(open("../../../Data/dnnARXIVTrainHistory.p","rb"))
	df = pd.DataFrame(nn_clf_history).plot(figsize=(10,5),title="ARXIV: DNN Model Training History")
	df.set_xlabel("Training Epoch")

	# Make prediction
	print("MAKING PREDICTIONS")
	# Y_pred = log_reg.predict(X_test)
	Y_pred = nn_clf.predict_classes(X_test)
	y_score = nn_clf.predict(X_test)

	# Compute ROC curve and ROC area for each class
	fpr = dict()
	tpr = dict()
	nn_threshold = dict()
	roc_auc = dict()

	for i in range(NUM_LABELS):
		fpr[i], tpr[i], _  = metrics.roc_curve(np.array(Y_test) , y_score[:,i] , pos_label=i)
		roc_auc[i] = metrics.auc(fpr[i], tpr[i])

	plt.figure()
	plt.plot([0, 1], [0, 1], 'r--')
	plt.xlim([0.0, 1.0])
	plt.ylim([0.0, 1.05])
	plt.xlabel('False Positive Rate')
	plt.ylabel('True Positive Rate')
	plt.title('ARXIV: DNN Model Receiver operating characteristic curve')

	# Plot of a ROC curve for a specific class
	for i in range(NUM_LABELS):
		plt.plot(fpr[i], tpr[i], label='ROC curve for label ' + str(i+1) + " " +list(LABELS.keys())[i]  +' (area = %0.2f)' % roc_auc[i])
		plt.legend(loc="lower right")

	plt.show()

	with open("../../../Data/ROC_Curves/DNN ARXIV.p","wb") as handle:
		curve = metrics.roc_curve(label_binarize(Y_test,classes=list(LABELS.values())).ravel(),y_score.ravel())
		auc = metrics.roc_auc_score(label_binarize(Y_test,classes=list(LABELS.values())),label_binarize(Y_pred,classes=list(LABELS.values())),average="micro")		
		pickle.dump((curve,auc),handle)

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
	
main()
