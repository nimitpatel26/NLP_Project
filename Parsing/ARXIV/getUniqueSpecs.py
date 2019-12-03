################################
#
# CMSC 473 Project
# getData.py
# Used to get the unique spec labels from the data
# created by the parse.py function
#
#################################

import pickle

allSpec = {}

def main():

	mainData = pickle.load(open("arXivSpec2grams.p", "rb"))
	for i in mainData:
		# If a specialization is not in allSpec keys, then add it
		for spec in i[1][0:1]:
			if(spec not in allSpec):
				allSpec[spec] = 0
			allSpec[spec] += 1	

	print("Number of Papers: " +str(len(mainData)))
	print(allSpec)

	
main()
