
################################
#
# CMSC 473 Project
# parse.py
# Used to parse the NSF dataset
# Outputs a pickle file
#
#################################

import os
import pickle

def main():
    fieldDict = {}
    mainData = []

    parts = 3

    # Used to count the number of files
    counter = 0

    mainPath = "/Users/nimitpatel/Documents/UMBC/Classes/Fall 2019/CMSC 473/Project/NSF Dataset/Part "
    # Run for four times for four parts
    # "~/path/Part 1" "~/path/Part 2" "~/path/Part 3" "~/path/Part 4"
    for i in range(1, parts + 1):
        for subdir, dirs, files in os.walk(mainPath + str(i)):
            for filename in files:
                filepath = subdir + os.sep + filename
                counter = counter + 1

                # If the file is a txt file
                if filepath.endswith(".txt"):
                    fieldFound = False
                    progRefFound = False
                    abstractFound = False
                    abstract = ""
                    tags = []
                    dataset = []
                    #print(filepath)

                    # If the file has invalid unicode characters, it will crash
                    # so errors='ignore'
                    f = open(filepath, "r", encoding='utf-8', errors='ignore')

                    # Go through the file line by line
                    for line in f:
                        if line[:13] == "Program Ref :":
                            progRefFound = True

                        # Get the field application
                        if (line[:13] == "Fld Applictn:" and not fieldFound) or (fieldFound and not progRefFound):
                            fieldFound = True
                            index = line[14:-1].strip()
                            tags.append(index)

                            # Tracking how many abstracts from each subject showed up
                            key = fieldDict.get(index)
                            if key == None:
                                fieldDict[index] = 1
                            else:
                                fieldDict[index] = key + 1

                        # Get the abstract
                        if abstractFound:
                            abstract = abstract + " " + line.strip()
                        if line[:13] == "Abstract    :":
                            abstractFound = True
                    
                    f.close()
                    dataset = [abstract.strip(), tags]
                    mainData.append(dataset)
    

    # Print statistics
    print("Files found: " + str(counter))
    print("Fields found:")
    for i in fieldDict:
        print(i + "\t" + str(fieldDict[i]))
    
    # Create the pickle file
    pickle.dump(mainData, open("data.p", "wb" ))
    
main()
