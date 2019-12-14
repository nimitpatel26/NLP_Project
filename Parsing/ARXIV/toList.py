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
	# list of data
	file1 = "arXivAllCategory.p"
	file2 = "arXivFirstCategory.p"
	file3 = "arXivSpecialization.p"
	mainDict = {}

	with (open(file1, "rb")) as openfile:
		data = pickle.load(openfile)
		for i in data:
			for j in data[i]:
				print(j)
				key = mainDict.get(j)
				if key == None:
					mainDict[j] = [i]
				else:
					mainDict[j].append(i)


	mainData = []
	for i in mainDict:
		dataSet = [i, mainDict[i]]
		mainData.append(dataSet)
	pickle.dump(mainData, open("arXivAll.p", "wb" ))
main()
