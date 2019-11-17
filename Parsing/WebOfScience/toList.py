################################
#
# CMSC 473 Project
# getData.py
# Used to get data from the pickle file
# created by the parse.py function
#
#################################

import pickle

def main():

	mainData = pickle.load(open("webOfScienceDomains.p", "rb"))
	mainDict = {}
	for i in mainData:
		key = mainDict.get(i)
		if key == None:
			mainDict[i] = [mainData[i][:-1]]
		else:
			mainDict[i].append(mainDict[i][:-1])

		# print(mainData[i])
		# print("\n------------------------------------\n")

	mainList = []
	for i in mainDict:
		dataSet = [i, mainDict[i]]
		mainList.append(dataSet)

	# for i in mainList:
	# 	print(i)
	# 	print("\n------------------------------------\n")
	pickle.dump(mainList, open("WOS.p", "wb" ))
	
main()
