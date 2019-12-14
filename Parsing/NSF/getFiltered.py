################################
#
# CMSC 473 Project
# getData.py
# Used to get the unique spec labels from the data
# created by the parse.py function
#
#################################

import pickle

allLabels = {}

def main():

	mainData = pickle.load(open("relabeledNSF.p", "rb"))

	dataFiltered = [[i[0],[label for label in i[1] if label != "UNK"]] for i in mainData if ( "".join(i[1]) != "UNK" and len(i[1]) > 0 )]

	with open("relabeledNSFfiltered.p","wb") as handle:
		pickle.dump(dataFiltered,handle)

	mainData = dataFiltered

	counter = 0

	for i in mainData:
		# If a specialization is not in allSpec keys, then add it
		for spec in i[1][0:1]:
			if(spec not in allLabels):
				allLabels[spec] = 0
			allLabels[spec] += 1

		counter+=1
			


	print("Number of Papers: " +str(len(mainData)))
	print(allLabels)

	
main()
