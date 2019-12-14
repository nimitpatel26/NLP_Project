################################
#
# CMSC 473 Project
# getData.py
# Used to get data from the pickle file
# created by the parse.py function
#
#################################

import pickle

classes = {}
badClassCount = 0
total = 0
badClasses = ["0000099   Other Applications NEC", "99        Other Sciences NEC", "null", "9900000     DATA NOT AVAILABLE", ""]



def addClasses(tmpClass):
	global badClasses
	global badClassCount
	global total
	isBadClass = True
	# tmp = False
	for i in tmpClass:
		key = classes.get(i)
		if key == None:
			classes[i] = 1
		else:
			classes[i] = key + 1
		if i not in badClasses:
			isBadClass = False
	if isBadClass == True:
		badClassCount = badClassCount + 1
	total = total + 1


def main():
	global badClasses
	global badClassCount
	global total
	badClassCount = 0
	# list of data
	mainData = pickle.load(open("data.p", "rb"))
	for i in mainData[1:]:
		addClasses(i[1])

	sum = 0
	for i in classes:
		isBadClass = False
		if i in badClasses:
			isBadClass = True
		print(i + "\t" + str(classes[i]) + "\t" + str(isBadClass))
		sum = sum + classes[i]
	print("Total = " + str(total))
	print("Bad Labels = " + str(badClassCount))
	print("% Bad Labels = " + str(round(badClassCount / total * 100, 2)) + "%")
main()
