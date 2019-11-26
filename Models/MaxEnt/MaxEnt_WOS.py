

import pickle
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

DATA = {}
FILES_PER_LABEL = {}
VOCAB = ["the", "of", "and", "to", "a", "in", "is", "for", "that", "with", "The", "on", "this", "are", "by", "In", "an", "as", "we", "be", "can", "which", "from", "This", "data", "using", "We", "results", "based", "has", "used", "have", "paper", "proposed", "also", "it", "such", "or", "been", "method", "our", "(C)", "these", "different", "paper,", "system", "not", "new", "at", "show", "their", "two", "Elsevier", "algorithm", "more", "performance", "between", "model", "analysis", "All", "its", "use", "into", "rights", "reserved.", "computer", "approach", "was", "A", "present", "study", "one", "other", "information", "However,", "than", "propose", "both", "To", "were", "time", "number", "most", "important", "design", "2016", "each", "provide", "systems", "but", "methods", "many", "order", "high", "all", "software", "Our", "well", "through", "only", "network", "problem", "when", "It", "large", "research", "some", "computing", "set", "process", "novel", "image", "algorithms", "compared", "applications", "learning", "developed", "processing", "several", "existing", "various", "over", "first", "development", "application", "work", "while", "how", "may", "applied", "efficient", "demonstrate", "including", "security", "distributed", "where", "three", "B.V.", "Ltd.", "parallel", "they", "provides", "techniques", "improve", "multiple", "will", "models", "experimental", "computational", "very", "complex", "solutions", "better", "programming", "key", "obtained", "presents", "then", "These", "features", "framework", "implementation", "any", "For", "support", "same", "data.", "real", "due", "without", "structure", "there", "solution", "As", "control", "accuracy", "under", "systems.", "about", "main", "significant", "during", "presented", "problems", "effective", "machine", "terms", "within", "technique", "potential", "need", "case", "Finally,", "further", "experiments", "implemented", "detection", "related", "structures", "studies", "allows", "efficiency", "available", "operating", "specific", "achieve", "possible", "quality", "out", "designed", "performed", "so", "traditional", "current", "images", "found", "among", "function", "single", "Furthermore,", "With", "up", "them", "tool", "field", "user", "Moreover,", "approaches", "called", "significantly", "system.", "simulation", "computation", "via", "given", "evaluate", "evaluation", "addition,", "identify", "test", "applications.", "level", "scheme", "able", "those", "because", "2017", "knowledge", "recent", "no", "common", "known", "networks", "could", "feature", "best", "dynamic", "widely", "technology", "become", "help", "engineering", "effectiveness", "higher", "management", "types", "reduce", "parameters", "solve", "optimization", "communication", "result", "uses", "make", "often", "standard", "required", "power", "shows", "By", "Therefore,", "even", "shown", "describe", "find", "classification", "study,", "improved", "low", "role", "simple", "linear", "develop", "general", "showed", "optimal", "small", "access", "highly", "being", "characteristics", "way", "local", "against", "database", "data,", "obtain", "generated", "method,", "good", "analyze", "range", "state-of-the-art", "properties", "associated", "relational", "human", "architecture", "distribution", "introduce", "perform", "useful", "tools", "proposes", "space", "conducted", "previous", "considered", "Inc.", "source", "state", "functions", "major", "An", "cost", "users", "increasing", "Based", "complexity", "memory", "vision", "point", "address", "investigate", "identified", "increase", "method.", "evaluated", "thus", "time.", "methods.", "environment", "if", "model.", "size", "systems,", "four", "task", "demonstrated", "focus", "comparison", "still", "requires", "future", "form", "databases", "numerical", "like", "suitable", "area", "target", "similar", "after", "symbolic", "detect", "part", "modeling", "practical", "especially", "problem.", "real-time", "energy", "students", "search", "Using", "nonlinear", "type", "2015", "should", "robust", "much", "determine", "ability", "work,", "3D", "code", "open", "hardware", "algorithm.", "makes", "issues", "challenges", "means", "providing", "model,", "corresponding", "article", "accurate", "domain", "limited", "mobile", "global", "effect", "rate", "achieved", "mechanism", "visual", "performance.", "strategy", "do", "provided", "requirements", "less", "analyzed", "values", "virtual", "challenging", "interaction", "generate", "input", "difficult", "aim", "random", "objective", "certain", "devices", "amount", "statistical", "critical", "resulting", "variety", "graphics", "fast", "platform", "changes", "discuss", "system,", "digital", "along", "value", "understanding", "overall", "allow", "introduced", "basic", "tasks", "resources", "integrated", "error", "storage", "across", "since", "spatial", "processes", "sets", "few", "analysis.", "second", "solving", "automatically", "does", "impact", "implement", "effectively", "require", "made", "efficiently", "bioinformatics", "language", "total", "Experimental", "context", "algorithms.", "behavior", "cannot", "lower", "time,", "conventional", "generation", "needs", "algorithm,", "tested", "basis", "compare", "program", "physical", "process.", "Results:", "patterns", "Wiley", "patients", "treatment", "had", "clinical", "disease", "Methods:", "increased", "=", "risk", "who", "health", "effects", "Conclusions:", "levels", "Background:", "age", "factors", "years", "group", "included", "reported", "findings", "review", "Conclusion:", "2", "patients.", "<", "evidence", "There", "suggest", "observed", "assessed", "1", "care", "association", "patient", "whether", "did", "symptoms", "chronic", "therapy", "assess", "women", "primary", "activity", "differences", "groups", "3", "diagnosis", "mean", "cell", "reduced", "decreased", "positive", "early", "factor", "healthy", "children", "response", "regression", "cells", "examined", "measured", "expression", "outcomes", "participants", "investigated", "revealed", "aimed", "treated", "Objective:", "respectively.", "Results", "greater", "prevalence", "Patients", "therapeutic", "following", "relationship", "Although", "medical", "95%", "patients,", "population", "blood", "study.", "rates", "likely", "ratio", "sample", "protein", "+/-", "examine", "efficacy", "mechanisms", "before", "body", "life", "cases", "inflammatory", "months", "stress", "individuals", "regarding", "cohort", "however,", "disease.", "(n", "presence", "negative", "diseases", "treatment.", "6", "12", "intervention", "4", "analyses", "Health", "(p", "baseline", "confidence", "report", "completed", "age,", "aged", "outcome", "functional", "received", "might", "social", "influence", "10", "normal", "severe", "disorder", "p", "collected", "drug", "difference", "followed", "Data", "per", "involved", "cancer", "either", "change", "acute", "incidence", "follow-up", "loss", "adverse", "period", "whereas", "serum", "assessment", "disorders", "measures", "brain", "scores", "score", "least", "reduction", "literature", "status", "skin", "diagnosed", "5", "groups.", "male", "After", "subjects", "correlated", "Published", "cause", "gene", "No", "weight", "according", "interventions", "induced", "history", "characterized", "years.", "long-term", "needed", "indicated", "years,", "associations", "female", "cognitive", "particularly", "infection", "Of", "strategies", "group.", "indicate", "days", "syndrome", "and/or", "population.", "controls", "independent", "remains", "would", "(P", "conditions", "Among", "receptor", "cross-sectional", "determined", "family", "bone", "disease,", "P", "index", "mass", "suggests", "exposure", "frequency", "underlying", "Study", "decrease", "underwent", "purpose", "active", "randomized", "prevention", "people", "interval", "include", "logistic", "CI", "pain", "poor", "correlation", "older", "play", "relevant", "men", "previously", "studied", "controlled", "severity", "having", "At", "diagnostic", "weeks", "affect", "affected", "survey", "statistically", "trials", "developing", "mental", "experience", "contribute", "improvement", "tissue", "importance", "imaging", "lead", "adult", "water", "construction", "building", "environmental", "surface", "material", "green", "smart", "harvesting", "rainwater", "solar", "soil", "project", "materials", "bridge", "pollution", "monitoring", "areas", "carried", "natural", "sustainable", "temperature", "geotextile", "sensing", "remote", "structural", "suspension", "tests", "thermal", "urban", "projects", "buildings", "maximum", "economic", "2014", "capacity", "flow", "production", "ambient", "average", "demand", "consumption", "aims", "technologies", "land", "Water", "climate", "activities", "impacts", "-", "concentration", "mechanical", "caused", "selected", "samples", "supply", "conditions.", "rainfall", "alternative", "mainly", "environment.", "appropriate", "great", "promising", "composite", "light", "industry", "layer", "employed", "series", "combined", "methodology", "combination", "considering", "Society", "degrees", "estimated", "sources", "sustainability", "enhance", "daily", "carbon", "element", "increases", "sensor", "growth", "wind", "load", "density", "measurements", "benefits", "concept", "reference", "estimate", "strength", "relative", "around", "established", "electrical", "signal", "circuit", "voltage", "controller", "output", "electric", "operational", "analog", "presented.", "circuits", "operation", "amplifier", "motor", "electricity", "filter", "identification", "PID", "CMOS", "stability", "simulations", "speed", "measurement", "representation", "feedback", "phase", "parameter", "gain", "mu", "generator", "transfer", "converter", "theoretical", "noise", "differential", "simulated", "derived", "verified", "mode", "m", "microcontroller", "components", "hybrid", "consists", "signals", "estimation", "dynamics", "paper.", "mathematical", "equivalent", "controllers", "state-space", "transmission", "tracking", "controller.", "adaptive", "graph", "proposed.", "device", "loop", "prototype", "advantages", "describes", "Simulation", "control.", "connected", "circuits.", "electronic", "wide", "example", "sensitivity", "drive", "external", "procedure", "switching", "direct", "coupled", "validated", "fuel", "matrix", "industrial", "equations", "grid", "constant", "passive", "DC", "network.", "results.", "renewable", "transient", "described", "tuning", "directly", "additional", "integration", "delay", "unit", "continuous", "theory", "verify", "fuzzy", "parameters.", "molecular", "chain", "blotting", "reaction", "genes", "genetic", "biology", "polymerase", "DNA", "detected", "expressed", "cells.", "acid", "Northern", "proteins", "Southern", "metabolism", "confirmed", "Here,", "species", "biological", "immune", "mRNA", "quantitative", "enzyme", "PCR", "RNA", "cellular", "cells,", "sequence", "vitro", "tumor", "metabolic", "transcription", "regulation", "assay", "reverse", "tissues", "isolated", "enzymes", "plant", "essential", "binding", "suggesting", "activation", "resistance", "vivo", "blotting.", "formation", "roles", "pathways", "genome", "signaling", "genomic", "pathway", "virus", "plants", "sequencing", "containing", "amino", "lines", "mice", "stem", "region", "suggested", "responses", "resulted", "sequences", "produced", "mouse", "individual", "plays", "differentiation", "assays", "rapid", "enhanced", "host", "concentrations", "bacterial", "cDNA", "Thus,", "regulatory", "Western", "interactions", "blotting,", "species.", "chemical", "transgenic", "metabolism.", "substrate", "inhibition", "recently", "T", "I", "Recent", "markers", "physiological", "addition", "(c)", "mutations", "acids", "encoding", "unique", "liver", "nuclear", "leading", "distinct", "five", "Here", "manufacturing", "fluid", "engine", "internal", "combustion", "heat", "pressure", "gas", "finite", "computer-aided", "hydraulic", "velocity", "calculated", "mechanics", "design.", "Engineering", "volume", "materials.", "boundary", "analytical", "initial", "machines", "University", "thermodynamic", "predict", "design,", "hydraulics", "advanced", "product", "thermodynamics", "air", "account", "geometry", "working", "Manufacturing", "equation", "flux", "detailed", "transport", "cycle", "discussed", "length", "discussed.", "engineering.", "channel", "vehicle", "variation", "torque", "fundamental", "larger", "three-dimensional", "free", "necessary", "materials,", "force", "shear", "attention", "emotional", "gender", "personality", "implications", "Participants", "psychological", "behavioral", "child", "depression", "perceived", "predicted", "psychiatric", "young", "relationships", "behaviors", "behavior.", "abuse", "disorders.", "sexual", "little", "explore", "violence", "toward", "adults", "affective", "(i.e.,", "(N", "depressive", "measure", "positively", "lack", "prenatal", "what", "While", "consistent", "eating", "public", "adolescents", "variables", "experiences", "focused", "relation", "anxiety", "aspects", "nonverbal", "attitudes", "media", "mediated", "strong", "neural", "prosocial", "perception", "understand", "community", "Findings", "towards", "yet", "(e.g.,", "violent", "antisocial", "education", "particular", "developmental", "development.", "schizophrenia", "others", "disorder.", "childhood", "mood", "behavior,", "Research", "perceptions", "examines", "prior", "emotion", "recognition"]
LABELS = {"CS":0, "Medical":1, "Civil":2, "ECE":3, "biochemistry":4, "MAE":5, "Psychology ":6}

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

	mainData = pickle.load(open("WOS.p", "rb"))
	# print(len(mainData))
	margin = len(mainData) * .2
	flip = 0
	X = []
	Y = []

	X_test = []
	Y_test = []
	counter = 0
	for i in mainData:
		feature = {}
		for j in VOCAB:
			feature[j] = 0
		
		sentence = i[0].split(" ")

		# countVocab(sentence, i[1][0])
		# print(i)
		# print("\n------------------------------------\n")
		# key = FILES_PER_LABEL.get(i[1][0])
		# if key == None:
		# 	FILES_PER_LABEL[i[1][0]] = 1
		# else:
		# 	FILES_PER_LABEL[i[1][0]] = key + 1
		for j in sentence:
			key = feature.get(j)
			if key != None:
				feature[j] = 1
		if flip % 5 != 0:
		# if counter < margin:
			X.append(list(feature.values()))
			Y.append(LABELS[i[1][0]])
		else:
			X_test.append(list(feature.values()))
			Y_test.append(LABELS[i[1][0]])

		counter = counter + 1
		flip = flip + 1

	# print(FILES_PER_LABEL)
	# counter = 0
	# print("{", end = "")
	# for i in FILES_PER_LABEL:
	# 	print ("\"" + i + "\":" + str(counter) + ", ", end = "")
	# 	counter = counter + 1
	# print("}")
	# exit()
	X = np.array(X)
	Y = np.array(Y)

	print(X)
	print(Y)

	X_test = np.array(X_test)
	Y_test = np.array(Y_test)

	log_reg = LogisticRegression(multi_class="multinomial",solver="lbfgs", C=1, max_iter=1000)

	# Fit the model
	print("FITTING THE DATA")
	log_reg.fit(X, Y)

	# Make prediction
	print("MAKING PREDICTIONS")
	Y_pred = log_reg.predict(X_test)


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



	
main()
