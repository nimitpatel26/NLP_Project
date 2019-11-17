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

	mainData = pickle.load(open("arXivAll.p", "rb"))
	for i in mainData:
		print(i)
		print("\n------------------------------------\n")

	
main()
