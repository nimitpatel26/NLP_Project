
############################
# Nimit Patel
# CMSC 473
# Assignment 3
# Question 7 Code
############################

import sys
import math
import pickle


trainFile = ["./en_ewt-ud-train.conllu", "es_gsd-ud-train.conllu", "fr_gsd-ud-train.conllu", "pt_gsd-ud-train.conllu", "zh_gsd-ud-train.conllu"]
trainFileLabels = ["ENG", "ESP", "FRE", "POR", "ZHO"]


# Hyperparameters
vocabRate = .9999


# Used by the program to store the data
countList = []
totalList = []
totalSentList = []
probList = []
priorProb = []

# Used by getPairs to add a sentence to the dictionary
def addToDict(dict, sent, total):
	for i in range(len(sent)):
		dictEntry = sent[i]

		ret = dict.get(dictEntry, -1)
		total = total + 1

		if ret == -1:
			dict[dictEntry] = 1 
		else:
			dict[dictEntry] = ret + 1

	return dict, total
    
# Returns a dictionary with count by a pair and total
def getPairs(fileName):

	fle = open(fileName)
	dict = {}
	currSent = ""
	total = 0
	totalSent = 0

	# Goes through line by line counting words
	# Counts how many times they appear
	# Each sentence is sent to the addToDict() function
	for line in fle:
		if line[0] != '#' and line[0] != '\n':
			tmp = line.split('\t')
			currSent = currSent + "%split%" + tmp[1]
		elif line[0] == '\n':
			# Send the sentence to addToDict function to get the number of pairs
			totalSent = totalSent + 1
			dict, total = addToDict(dict, currSent.split("%split%")[1:], total)
			currSent = ""
	fle.close


	return dict, total, totalSent

def calcProb(sent, model):
	# print("---------------------------------")
	prob = 1
	for i in sent:
		# print(i)
		ret = model.get(i, -1)
		probTmp = ret
		if ret == -1:
			# print(i + "\t<UNK>")
			# prob = prob * float(model["<UNK>"])
			prob = prob + math.log(float(model["<UNK>"]), math.e)
		else:
			# print(i + "\tFound")
			# prob = prob * probTmp
			prob = prob + math.log(probTmp, math.e)
	# print("---------------------------------")
	return prob


def generateModel():
	global trainFile
	global trainFileLabels

	global vocabRate



	global countList
	global totalList
	global totalSentList
	global probList
	global priorProb

	for i in trainFile:
		dictTmp, totalTmp, totalSentTmp = getPairs(i)

		# Calculates the probability
		probTmp = {}
		for j in dictTmp:
			prob = float(dictTmp[j]) / float(totalTmp)
			probTmp[j] = prob

		oovCount = int(len(dictTmp) * (1 - vocabRate))
		sortedRelation = sorted(dictTmp.items(), key = lambda kv:(kv[1], kv[0]), reverse = False)
		dictTmp["<UNK>"] = 0
		for j in range(oovCount):
			tmpCount = dictTmp[sortedRelation[j][0]]
			del dictTmp[sortedRelation[j][0]]
			dictTmp["<UNK>"] = dictTmp["<UNK>"] + tmpCount
		probTmp["<UNK>"] = float(dictTmp["<UNK>"]) / float(totalTmp)

		countList.append(dictTmp)
		totalList.append(totalTmp)
		totalSentList.append(totalSentTmp)
		probList.append(probTmp)

	totalSentences = 0
	for i in totalSentList:
		totalSentences = totalSentences + i

	for i in range(len(totalSentList)):
		priorProb.append(totalSentList[i] / totalSentences)
		
def main():
	global trainFileLabels
	global probList
	global priorProb

	testSent = "Essa observação foi motivada por notícias de que Tim Morrison, principal autoridade do Conselho de Segurança Nacional da Casa Branca para a Eurásia, deixará o cargo após testemunhar que outros nomeados por Trump disseram aos ucranianos que não receberiam assistência de segurança a menos que iniciassem as investigações de corrupção que Trump procurava. A saída de Morrison, dias após outro funcionário do NSC ter incendiado os aliados de Trump por causa de seu testemunho, continua uma tendência de funcionários do governo serem envolvidos pelo escândalo, enquanto aliados da Otan tentam manter uma frente unida com o presidente ucraniano Volodymyr Zelensky contra a Rússia."
	generateModel()

	max = -99999999
	maxIndex = -1

	for i in range(len(trainFile)):
		prob = calcProb(testSent.split(" "), probList[i])
		# prob = math.log(prob, math.e) + math.log(priorProb[i], math.e)
		prob = prob + math.log(priorProb[i], math.e)
		print (trainFileLabels[i] + "\t" + str(prob))
		if prob > max:
			max = prob
			maxIndex = i
	
	print("Based on the model, the sentence \"" + testSent + "\" is probably in " + trainFileLabels[maxIndex] + ".")


main()
