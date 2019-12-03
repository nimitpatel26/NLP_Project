

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

VOCAB = ["$", "{", "}", "we", "In", "show", "prove", "n", "A", "space", "some", "number", "group", "if", "G", "case", "all", "where", "function", "''", "finite", "1", "'s", "set", ":", "result", "\\mathbb", "functions", "over", "[", "2", "]", "any", "give", "class", "given", "equation", "general", "X", "when", "then", "solutions", "linear", "0", "particular", "consider", "algebra", "R", "spaces", "p", "the", "of", ".", ",", "and", "a", "in", "to", "is", ")", "(", "for", "with", "that", "We", "The", "are", "by", "as", "at", "from", "be", "an", "which", "can", "model", "field", "quantum", "have", "between", "results", "also", "two", "energy", "This", "these", "or", "not", "theory", "mass", "study", "has", "present", "find", "system", "magnetic", "phase", "than", "state", "one", "equations", "nonlinear", "chaotic", "periodic", "integrable", "solitons", "chaos", "soliton", "hierarchy", "turbulence", "orbits", "synchronization", "Lyapunov", "bifurcation", "exponents", "unstable", "flows", "exponent", "semiclassical", "oscillators", "Lax", "turbulent", "transformations", "Toda", "nonlinearity", "lattices", "integrability", "fractal", "trajectories", "billiard", "KdV", "bifurcations", "attractor", "hierarchies", "forcing", "KP", "intermittency", "solitary", "dissipative", "nonlocal", "cascade", "Bethe", "attractors", "front", "convection", "billiards", "Painlev\\", "Reynolds", "perturbed", "oscillatory", "population", "cell", "protein", "cells", "species", "biological", "DNA", "brain", "gene", "proteins", "patterns", "activity", "neurons", "growth", "evolutionary", "genetic", "functional", "response", "sequences", "populations", "expression", "genes", "individual", "disease", "mechanisms", "individuals", "mathematical", "binding", "cancer", "structural", "reaction", "connectivity", "fitness", "cellular", "genome", "folding", "biology", "membrane", "mutations", "RNA", "neuronal", "mutation", 
"tissue", "quantitative", "phylogenetic", "regulatory", "sites", "epidemic", "diversity", "synaptic", "on", "this", "paper", "problem", "our", "algorithm", "network", "it", "such", "using", "proposed", "based", "performance", "approach", "time", "information", "new", "networks", "propose", "used", "systems", "algorithms", "Our", "more", "each", "use", "work", "framework", "problems", "image", "graph", "provide", "how", "To", "neural", "applications", "training", "novel", "However", "optimal", "most", "design", "features", "many", "demonstrate", "complexity", "deep", "while", "images", "techniques", "data", "models", "learning", 
"method", "methods", "distribution", "analysis", "estimation", "regression", "process", "Bayesian", "inference", "sample", "statistical", "estimator", "distributions", "classification", "variables", "test", "error", "Gaussian", "probability", "machine", "estimators", "prediction", "selection", "likelihood", "sampling", "estimate", "samples", "often", "procedure", "studies", "sparse", "prior", "empirical", "simulation", "develop", "noise", "latent", "clustering", "loss", "posterior", "gradient", "setting", "mean", "variable", "robust", "Markov", "modeling", "market", "price", "risk", "financial", "volatility", "stock", "stochastic", "markets", "prices", "portfolio", "pricing", "trading", "economic", "asset", "returns", "series", "options", "option", "strategy", "measures", "strategies", "agents", "impact", "investment", "utility", "wealth", "stocks", "return", "assets", "costs", "capital", "default", "countries", "trade", "credit", "hedging", "transaction", "European", "income", "Brownian", "portfolios", "economy", "firms", "crisis", "investors", "finance", "&", "martingale", "implied", "arbitrage", "signal", "speech", "signals", "imaging", "reconstruction", "audio", "conventional", "speaker", "acoustic", "MIMO", "filter", "sound", "music", "localization", "antenna", "communications", "MRI", "ASR", "EEG", "beamforming", "radar", "RF", "voice", "5G", "mmWave", "UAV", "diagnosis", "filters", "CT", "DNN", "acquisition", "denoising", "SNR", "waveform", "speakers", "UAVs", "dB", "emotion", "artifacts", "ECG", "acquired", "MR", "NOMA", "precoding", "aerial", "reconstruct", "musical", "signal-to-noise", "learning-based", "battery", "quantile", "heterogeneity", "preferences", "panel", "equilibria", "endogenous", "unobserved", "welfare", "preference", "econometric", "instrumental", "consumers", "counterfactual", "travel", "IV", "regressors", "strategic", "beliefs", "regional", "wage", 
"seller", "households", "regressions", "logit", "cross-sectional", "incentives", "sieve", "GMM", "auctions", "econometrics", "participation", "loadings", "household", "EV", "disparity", "sender", "voters", "regressor", "heteroskedasticity", "Shapley", "rebalancing", "democracy", "2SLS", "agricultural", "women", "labour", "\\texttt", "panels", "willingness", "VB", ", we", ") $", "of a", "$ .", ". In", "$ ,", "} $", "of $", "show that", "$ is", "In this", "is a", "this paper", "$ and", "and $", "to a", "} (", "number of", "prove that", "is the", "\\mathbb {", "$ (", "We show", "We also", ", $", "we show", "on a", "of this", "We prove", "for a", "class of", "terms of", "in $", "in terms", "$ of", ", where", "we prove", "$ G", "G $", "} )", "set of", ". As", "\\mathcal {", "study the", "$ n", "n $", "paper we", "$ \\mathbb", "it is", "on $", "of the", "in the", ". The", ". We", "to the", ", and", ", the", "for the", "on the", "and the", "that the", "with the", "can be", ". This", "from the", ", which", "in a", "by the", "with a", ") .", "to be", "at the", ") ,", "as a", ", in", "find that", ". It", ". A", "between the", "due to", ". For", "by a", 
"as the", "We present", ") and", "the $", ". These", "We find", "and a", "properties of", "^ {", "which is", ", with", "It is", "as well", "of these", "has been", ", but", "In the", "well as", "dynamics of", "equation .", "equations .", "the dynamics", "phase space", "dynamics .", "numerical simulations", "dynamical systems", "systems with", "equation ,", "stability of", "the nonlinear", "periodic orbits", "of motion", "a system", "of nonlinear", "equation is", "the chaotic", "the stability", "equations are", "equations for", "solutions are", "chaotic systems", "of chaotic", "Lyapunov exponents", "dynamical system", "the scaling", "the semiclassical", "integrable systems", "of coupled", "a nonlinear", "Lyapunov exponent", "system with", "systems are", "cellular automata", "initial conditions", "dynamics is", "dynamics ,", "the Hamiltonian", "chaos .", "hierarchy .", "equation and", "the soliton", "solitons in", "Painlev\\ '", "statistics of", "of integrable", "random matrix", "nonlinear Schr\\", "Hamiltonian systems", "model of", "the population", "the brain", "The model", "gene expression", "changes in", "cells .", "the cell", "of protein", "of biological", "of DNA", "the protein", "response to", "proteins .", "species .", "a population", "cells ,", "of proteins", "the human", "of gene", "emergence of", "patterns of", "of neurons", "interactions between", "levels of", "network of", "of cells", "populations .", "of neural", "neurons .", "of human", "networks of", "population .", "the emergence", "of species", "mathematical model", "the DNA", "of individual", "of cell", "steady state", "involved in", "population size", "biological systems", "amino acid", "proteins ,", "amino acids", "the genetic", "species ,", "sequences .", "the evolutionary", "based on", "paper ,", ". Our", ", a", "such as", "we propose", "propose a", ". However", ". To", "However ,", "This paper", "the problem", "a novel", ", it", "of our", "problem of", "performance of", "O (", "we present", "this work", "order to", ", such", "the performance", ", our", "that are", "use of", "the network", "that our", "that can", "networks .", "able to", "work ,", "problem .", "a set", "algorithm for", ", to", "in order", "problem ,", "i.e. ,", "results show", "the best", "$ O", "approach to", "time .", "this problem", "our approach", "algorithm is", "used in", ", an", "network .", "data .", "We propose", "machine learning", "the data", "models .", "the model", "neural networks", "data ,", "method is", "Monte Carlo", "methods .", "models ,", "applied to", "demonstrate the", "our method", "estimation of", "method for", "to estimate", "of data", "approach is", "data and", "methods for", "data sets", "distribution .", "develop a", "the underlying", "method to", "learning .", "real data", "the sample", "methods ,", "of interest", "estimate the", "maximum likelihood", "in many", "the training", "deep neural", "we develop", "datasets .", "sample size", "distributions .", "data from", "methods are", "reinforcement learning", "method ,", "models are", "variables .", "analysis .", "data is", "learning ,", "the market", "time series", "the optimal", "the price", "markets .", "of financial", "market .", "stock market", "financial markets", "process .", "stochastic volatility", "the distribution", "risk measures", "the value", "the stock", "the volatility", "the financial", "financial market", "price of", "of market", "the risk", "transaction costs", "power law", "in financial", "Brownian motion", "impact of", "interest rate", "processes .", "of risk", "prices .", "stock price", "systemic risk", "value function", "a stochastic", "markets ,", "market ,", "the impact", "risk .", "of time", "option pricing", "of stock", "implied volatility", "volatility .", "process ,", "an optimal", "a financial", "series .", "the pricing", "the portfolio", "prices of", "the proposed", "neural network", "The proposed", "proposed method", "deep learning", "is proposed", "images .", "network (", "convolutional neural", "speech recognition", "a deep", "signal processing", ". Simulation", "signals .", "the signal", "the conventional", "Simulation results", "error rate", "simulation results", "CNN )", "( CNN", ") based", "channel estimation", ") systems", "signal .", "ratio (", "% and", "The performance", "communication systems", "base station", "rate (", "is trained", "Neural Network", "is evaluated", "of speech", "multiple access", "MIMO )", "image quality", "( MIMO", "speech enhancement", "and power", "the receiver", "power consumption", "massive MIMO", "wireless communication", "signals ,", "signal-to-noise ratio", "power system", "noise ,", "generative adversarial", "treatment effects", "treatment effect", "panel data", "average treatment", "quantile regression", "an empirical", "and inference", "empirical application", "fixed effects", "This study", "the treatment", "inference on", "test for", "literature on", "discrete choice", "in economics", "instrumental variables", "Nash equilibrium", "standard errors", "unobserved heterogeneity", "asymptotic distribution", "estimators and", "heterogeneity in", "causal effects", "of treatment", "inference methods", "I show", "social welfare", "confidence bands", "paper develops", "the United", ". Monte", "economics and", "game theory", "sample properties", "conditional on", "economics .", "equilibrium ,", "preferences .", "the identified", "data models", "propensity score", "2018 )", "identified set", "instrumental variable", "finite samples", "heterogeneity .", "bias .", "the players", "Nash equilibria"]
LABELS = {'math': 1, 'physics': 2, 'nlin': 3, 'q-bio': 4,
          'cs': 5, 'stat': 6, 'q-fin': 7, 'econ': 8, 'eess': 9}

# def countVocab(sentence, label):
# 	wordsAdded = []
# 	for i in sentence:
# 		key = DATA.get(label)
# 		if key == None:
# 			DATA[label] = {i:1}
# 			wordsAdded.append(i)
# 		else:
# 			keyVocab = key.get(i)
# 			if keyVocab == None:
# 				key[i] = 1
# 			elif i not in wordsAdded:
# 				key[i] = keyVocab + 1
# 				wordsAdded.append(i)

# def getData():
# 	None

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

featureN = [1,2]

def abstractFeatureLabels(abstracts):

	global featureN
	global featureDict
	global LABELS

	X = dok_matrix((len(abstracts),len(VOCAB)))
	# Y =  [ [ 0 for i in range(0,1) ] for j in range(0,len(abstracts)) ]
	Y = []

	# a = some abstract number
	
	for a in range(0,len(abstracts)):

		sentence = abstracts[a][0].split(" ")

		# j = some token in the abstract

		for n in featureN:

			for i in range((len(sentence)-n+1)):

				gram = " ".join(sentence[i:i+n])

				if gram in featureDict:

					X[a,featureDict[gram]] = 1

				del gram

				gram = None

		del sentence

		sentence = None

		Y.append(LABELS[abstracts[a][1][0]])

	del abstracts

	abstracts = None

	return X,Y

# Instead of creating 20 data splits, create a split for each abstract in the data
def featureArraysNSplits(data):

	argtuples = []

	for a in data:
		argtuples.append((a[0],a[1]))

	abstractParsingPool = Pool(processes=10)
	
	map = abstractParsingPool.map_async(abstractFeatureLabel,argtuples)

	abstractParsingPool.close()

	abstractParsingPool.join()

	del argtuples
	
	argtuples = None
	
	res = map.get(timeout=0)

	return res


def main():

	mp.set_start_method("spawn")

	start = time.time()

	mainData = pickle.load(open("arXivSpecMergedTokens.p", "rb"))

	# split the data array into lists of tuples, each 1/20th the size of the original data

	argtuples20 = []

	for a in range(0,20):

		argtuples20.append(list([]))

	for a in range(0,len(mainData)):

		argtuples20[a%20].append((mainData[a][0],mainData[a][1]))

	del mainData

	mainData = None

	# each tuple list will get a process mapped to it, total of 20 processes in the pool

	abstractParsingPool = Pool(processes=10)

	map = abstractParsingPool.map_async(abstractFeatureLabels,argtuples20)

	abstractParsingPool.close()

	abstractParsingPool.join()

	# get the results from the processes that processed the data
	
	res = map.get(timeout=0)

	del argtuples20

	argtuples20 = None

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
