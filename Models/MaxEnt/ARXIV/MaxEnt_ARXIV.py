

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


DATA = {}
FILES_PER_LABEL = {}
VOCAB = ["the", "of", "and", "to", "a", "in", "is", "for", "that", "with", "The", "on", "this", "are", "by", "In", "an", "as", "we", "be", "can", "which", "from", "This", "data", "using", "We", "results", "based", "has", "used", "have", "paper", "proposed", "also", "it", "such", "or", "been", "method", "our", "(C)", "these", "different", "paper,", "system", "not", "new", "at", "show", "their", "two", "Elsevier", "algorithm", "more", "performance", "between", "model", "analysis", "All", "its", "use", "into", "rights", "reserved.", "computer", "approach", "was", "A", "present", "study", "one", "other", "information", "However,", "than", "propose", "both", "To", "were", "time", "number", "most", "important", "design", "2016", "each", "provide", "systems", "but", "methods", "many", "order", "high", "all", "software", "Our", "well", "through", "only", "network", "problem", "when", "It", "large", "research", "some", "computing", "set", "process", "novel", "image", "algorithms", "compared", "applications", "learning", "developed", "processing", "several", "existing", "various", "over", "first", "development", "application", "work", "while", "how", "may", "applied", "efficient", "demonstrate", "including", "security", "distributed", "where", "three", "B.V.", "Ltd.", "parallel", "they", "provides", "techniques", "improve", "multiple", "will", "models", "experimental", "computational", "very", "complex", "solutions", "better", "programming", "key", "obtained", "presents", "then", "These", "features", "framework", "implementation", "any", "For", "support", "same", "data.", "real", "due", "without", "structure", "there", "solution", "As", "control", "accuracy", "under", "systems.", "about", "main", "significant", "during", "presented", "problems", "effective", "machine", "terms", "within", "technique", "potential", "need", "case", "Finally,", "further", "experiments", "implemented", "detection", "related", "structures", "studies", "allows", "efficiency", "available", "operating", "specific", "achieve", "possible", "quality", "out", "designed", "performed", "so", "traditional", "current", "images", "found", "among", "function", "single", "Furthermore,", "With", "up", "them", "tool", "field", "user", "Moreover,", "approaches", "called", "significantly", "system.", "simulation", "computation", "via", "given", "evaluate", "evaluation", "addition,", "identify", "test", "applications.", "level", "scheme", "able", "those", "because", "2017", "knowledge", "recent", "no", "common", "known", "networks", "could", "feature", "best", "dynamic", "widely", "technology", "become", "help", "engineering", "effectiveness", "higher", "management", "types", "reduce", "parameters", "solve", "optimization", "communication", "result", "uses", "make", "often", "standard", "required", "power", "shows", "By", "Therefore,", "even", "shown", "describe", "find", "classification", "study,", "improved", "low", "role", "simple", "linear", "develop", "general", "showed", "optimal", "small", "access", "highly", "being", "characteristics", "way", "local", "against", "database", "data,", "obtain", "generated", "method,", "good", "analyze", "range", "state-of-the-art", "properties", "associated", "relational", "human", "architecture", "distribution", "introduce", "perform", "useful", "tools", "proposes", "space", "conducted", "previous", "considered", "Inc.", "source", "state", "functions", "major", "An", "cost", "users", "increasing", "Based", "complexity", "memory", "vision", "point", "address", "investigate", "identified", "increase", "method.", "evaluated", "thus", "time.", "methods.", "environment", "if", "model.", "size", "systems,", "four", "task", "demonstrated", "focus", "comparison", "still", "requires", "future", "form", "databases", "numerical", "like", "suitable", "area", "target", "similar", "after", "symbolic", "detect", "part", "modeling", "practical", "especially", "problem.", "real-time", "energy", "students", "search", "Using", "nonlinear", "type", "2015", "should", "robust", "much", "determine", "ability", "work,", "3D", "code", "open", "hardware", "algorithm.", "makes", "issues", "challenges", "means", "providing", "model,", "corresponding", "article", "accurate", "domain", "limited", "mobile", "global", "effect", "rate", "achieved", "mechanism", "visual", "performance.", "strategy", "do", "provided", "requirements", "less", "analyzed", "values", "virtual", "challenging", "interaction", "generate", "input", "difficult", "aim", "random", "objective", "certain", "devices", "amount", "statistical", "critical", "resulting", "variety", "graphics", "fast", "platform", "changes", "discuss", "system,", "digital", "along", "value", "understanding", "overall", "allow", "introduced", "basic", "tasks", "resources", "integrated", "error", "storage", "across", "since", "spatial", "processes", "sets", "few", "analysis.", "second", "solving", "automatically", "does", "impact", "implement", "effectively", "require", "made", "efficiently", "bioinformatics", "language", "total", "Experimental", "context", "algorithms.", "behavior", "cannot", "lower", "time,", "conventional", "generation", "needs", "algorithm,", "tested", "basis", "compare", "program", "physical", "process.", "Results:", "patterns", "Wiley", "patients", "treatment", "had", "clinical", "disease", "Methods:", "increased", "=", "risk", "who", "health", "effects", "Conclusions:", "levels", "Background:", "age", "factors", "years", "group", "included", "reported", "findings", "review", "Conclusion:", "2", "patients.", "<", "evidence", "There", "suggest", "observed", "assessed", "1", "care", "association", "patient", "whether", "did", "symptoms", "chronic", "therapy", "assess", "women", "primary", "activity", "differences", "groups", "3", "diagnosis", "mean", "cell", "reduced", "decreased", "positive", "early", "factor", "healthy", "children", "response", "regression", "cells", "examined", "measured", "expression", "outcomes", "participants", "investigated", "revealed", "aimed", "treated", "Objective:", "respectively.", "Results", "greater", "prevalence", "Patients", "therapeutic", "following", "relationship", "Although", "medical", "95%", "patients,", "population", "blood", "study.", "rates", "likely", "ratio", "sample", "protein", "+/-", "examine", "efficacy", "mechanisms", "before", "body", "life", "cases", "inflammatory", "months", "stress", "individuals", "regarding", "cohort", "however,", "disease.", "(n", "presence", "negative", "diseases", "treatment.", "6", "12", "intervention", "4", "analyses", "Health", "(p", "baseline", "confidence", "report", "completed", "age,", "aged", "outcome", "functional", "received", "might", "social", "influence", "10", "normal", "severe", "disorder", "p", "collected", "drug", "difference", "followed", "Data", "per", "involved", "cancer", "either", "change", "acute", "incidence", "follow-up", "loss", "adverse", "period", "whereas", "serum", "assessment", "disorders", "measures", "brain", "scores", "score", "least", "reduction", "literature", "status", "skin", "diagnosed", "5", "groups.", "male", "After", "subjects", "correlated", "Published", "cause", "gene", "No", "weight", "according", "interventions", "induced", "history", "characterized", "years.", "long-term", "needed", "indicated", "years,", "associations", "female", "cognitive", "particularly", "infection", "Of", "strategies", "group.", "indicate", "days", "syndrome", "and/or", "population.", "controls", "independent", "remains", "would", "(P", "conditions", "Among", "receptor", "cross-sectional", "determined", "family", "bone", "disease,", "P", "index", "mass", "suggests", "exposure", "frequency", "underlying", "Study", "decrease", "underwent", "purpose", "active", "randomized", "prevention", "people", "interval", "include", "logistic", "CI", "pain", "poor", "correlation", "older", "play", "relevant", "men", "previously", "studied", "controlled", "severity", "having", "At", "diagnostic", "weeks", "affect", "affected", "survey", "statistically", "trials", "developing", "mental", "experience", "contribute", "improvement", "tissue", "importance", "imaging", "lead", "adult", "water", "construction", "building", "environmental", "surface", "material", "green", "smart", "harvesting", "rainwater", "solar", "soil", "project", "materials", "bridge", "pollution", "monitoring", "areas", "carried", "natural", "sustainable", "temperature", "geotextile", "sensing", "remote", "structural", "suspension", "tests", "thermal", "urban", "projects", "buildings", "maximum", "economic", "2014", "capacity", "flow", "production", "ambient", "average", "demand", "consumption", "aims", "technologies", "land", "Water", "climate", "activities", "impacts", "-", "concentration", "mechanical", "caused", "selected", "samples", "supply", "conditions.", "rainfall", "alternative", "mainly", "environment.", "appropriate", "great", "promising", "composite", "light", "industry", "layer", "employed", "series", "combined", "methodology", "combination", "considering", "Society", "degrees", "estimated", "sources", "sustainability", "enhance", "daily", "carbon", "element", "increases", "sensor", "growth", "wind", "load", "density", "measurements", "benefits", "concept", "reference", "estimate", "strength", "relative", "around", "established", "electrical", "signal", "circuit", "voltage", "controller", "output", "electric", "operational", "analog", "presented.", "circuits", "operation", "amplifier", "motor", "electricity", "filter", "identification", "PID", "CMOS", "stability", "simulations", "speed", "measurement", "representation", "feedback", "phase", "parameter", "gain", "mu", "generator", "transfer", "converter", "theoretical", "noise", "differential", "simulated", "derived", "verified", "mode", "m", "microcontroller", "components", "hybrid", "consists", "signals", "estimation", "dynamics", "paper.", "mathematical", "equivalent", "controllers", "state-space", "transmission", "tracking", "controller.", "adaptive", "graph", "proposed.", "device", "loop", "prototype", "advantages", "describes", "Simulation", "control.", "connected", "circuits.", "electronic", "wide", "example", "sensitivity", "drive", "external", "procedure", "switching", "direct", "coupled", "validated", "fuel", "matrix", "industrial", "equations", "grid", "constant", "passive", "DC", "network.", "results.", "renewable", "transient", "described", "tuning", "directly", "additional", "integration", "delay", "unit", "continuous", "theory", "verify", "fuzzy", "parameters.", "molecular", "chain", "blotting", "reaction", "genes", "genetic", "biology", "polymerase", "DNA", "detected", "expressed", "cells.", "acid", "Northern", "proteins", "Southern", "metabolism", "confirmed", "Here,", "species", "biological", "immune", "mRNA", "quantitative", "enzyme", "PCR", "RNA", "cellular", "cells,", "sequence", "vitro", "tumor", "metabolic", "transcription", "regulation", "assay", "reverse", "tissues", "isolated", "enzymes", "plant", "essential", "binding", "suggesting", "activation", "resistance", "vivo", "blotting.", "formation", "roles", "pathways", "genome", "signaling", "genomic", "pathway", "virus", "plants", "sequencing", "containing", "amino", "lines", "mice", "stem", "region", "suggested", "responses", "resulted", "sequences", "produced", "mouse", "individual", "plays", "differentiation", "assays", "rapid", "enhanced", "host", "concentrations", "bacterial", "cDNA", "Thus,", "regulatory", "Western", "interactions", "blotting,", "species.", "chemical", "transgenic", "metabolism.", "substrate", "inhibition", "recently", "T", "I", "Recent", "markers", "physiological", "addition", "(c)", "mutations", "acids", "encoding", "unique", "liver", "nuclear", "leading", "distinct", "five", "Here", "manufacturing", "fluid", "engine", "internal", "combustion", "heat", "pressure", "gas", "finite", "computer-aided", "hydraulic", "velocity", "calculated", "mechanics", "design.", "Engineering", "volume", "materials.", "boundary", "analytical", "initial", "machines", "University", "thermodynamic", "predict", "design,", "hydraulics", "advanced", "product", "thermodynamics", "air", "account", "geometry", "working", "Manufacturing", "equation", "flux", "detailed", "transport", "cycle", "discussed", "length", "discussed.", "engineering.", "channel", "vehicle", "variation", "torque", "fundamental", "larger", "three-dimensional", "free", "necessary", "materials,", "force", "shear", "attention", "emotional", "gender", "personality", "implications", "Participants", "psychological", "behavioral", "child", "depression", "perceived", "predicted", "psychiatric", "young", "relationships", "behaviors", "behavior.", "abuse", "disorders.", "sexual", "little", "explore", "violence", "toward", "adults", "affective", "(i.e.,", "(N", "depressive", "measure", "positively", "lack", "prenatal", "what", "While", "consistent", "eating", "public", "adolescents", "variables", "experiences", "focused", "relation", "anxiety", "aspects", "nonverbal", "attitudes", "media", "mediated", "strong", "neural", "prosocial", "perception", "understand", "community", "Findings", "towards", "yet", "(e.g.,", "violent", "antisocial", "education", "particular", "developmental", "development.", "schizophrenia", "others", "disorder.", "childhood", "mood", "behavior,", "Research", "perceptions", "examines", "prior", "emotion", "recognition"]
LABELS = {'math': 1, 'physics': 2, 'nlin': 3, 'q-bio': 4, 'cs': 5, 'stat': 6, 'q-fin': 7, 'econ': 8, 'eess': 9}

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

def abstractFeatureLabels(abstracts):

	global featureDict
	global LABELS

	X = dok_matrix((len(abstracts),len(VOCAB)))
	# Y =  [ [ 0 for i in range(0,1) ] for j in range(0,len(abstracts)) ]
	Y = []
	
	for a in range(0,len(abstracts)):

		for j in abstracts[a][0]:

			if j in featureDict:

				X[a,featureDict[j]] = 1

		# Y[a][0] = (LABELS[abstracts[a][1][0]])
		Y.append(LABELS[abstracts[a][1][0]])

	return X,Y

# Instead of creating 20 data splits, create a split for each abstract in the data
def featureArraysNSplits(data):

	mp.set_start_method("spawn")

	argtuples = []

	for a in data:
		argtuples.append((a[0],a[1]))

	abstractParsingPool = Pool()
	
	map = abstractParsingPool.map_async(abstractFeatureLabel,argtuples)

	abstractParsingPool.close()

	abstractParsingPool.join()
	
	res = map.get(timeout=0)

	return res


def main():

	start = time.time()

	mainData = pickle.load(open("arXivSpec.p", "rb"))

	# split the data array into lists of tuples, each 1/20th the size of the original data

	argtuples20 = []

	for a in range(0,20):

		argtuples20.append(list([]))

	for a in range(0,len(mainData)):

		argtuples20[a%20].append((mainData[a][0],mainData[a][1]))

	# each tuple list will get a process mapped to it, total of 20 processes in the pool

	abstractParsingPool = Pool()

	map = abstractParsingPool.map_async(abstractFeatureLabels,argtuples20)

	abstractParsingPool.close()

	abstractParsingPool.join()

	# get the results from the processes that processed the data
	
	res = map.get(timeout=0)

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

	print(Y_pred.tolist())

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
