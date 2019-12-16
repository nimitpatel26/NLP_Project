<<<<<<< HEAD

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

global separateList
global separateListFix

def fixAbstract(abstract):
    global separateList
    global separateListFix
    newAbstract = ""
    marker = 0
    stopIndicies = []
    for i in range(len(abstract)):
        if abstract[i] in separateList:
            stopIndicies.append(i)

    for i in stopIndicies:
        j = separateList.index(abstract[i])
        newAbstract = newAbstract + abstract[marker:i] + separateListFix[j]
        marker = i + 1

    newAbstract = newAbstract + abstract[marker:]
    return newAbstract

def main():
    global separateList
    global separateListFix
    separateList = ["(", ")", "{", "}", "[", "]", ",", "<", ">", ";", ":", "-", ", " ,".", "!", "?"]
    separateListFix = ["( ", " )", "{ ", " }", "[ ", " ]", " ,", "< ", " >", " ;", " :", " - ", " , " ," .", " !", " ?"]
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
                    dataset = [fixAbstract(abstract.strip()), tags]
                    mainData.append(dataset)
    

    # Print statistics
    print("Files found: " + str(counter))
    print("Fields found:")
    for i in fieldDict:
        print(i + "\t" + str(fieldDict[i]))
    
    # Create the pickle file
    pickle.dump(mainData, open("data.p", "wb" ))
    
main()
=======

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

classes = {}
badClassCount = 0
total = 0
badClasses = ["0000099   Other Applications NEC","0000099   Other Applications NEC", "99        Other Sciences NEC", "null", "9900000     DATA NOT AVAILABLE", ""]
labels = ["0000099   Other Applications NEC", "99        Other Sciences NEC", "59        Engineering NEC", "21        Mathematics", "49        Environmental NEC", "61        Life Science Biological", "12        Chemistry", "90        Other Sciences", "31        Computer Science & Engineering", "13        Physics", "89        Social Sciences NEC", "19        Physical Sciences NEC", "42        Geological Sciences", "88        Geography", "40        Environmental Sciences", "60        Life Sciences", "81        Anthropology", "10        Physical Sciences", "0105000   Manpower & Training", "98        Multi-Disciplinary", "0101000   Curriculum Development", "55        Engineering-Electrical", "50        Engineering", "91        Science Technology Assess", "11        Astronomy", "41        Atmospheric Sciences", "43        Biological Oceanography", "20        Mathematics", "79        Psychology Sciences NEC", "80        Social Sciences", "69        Life Sciences NEC", "56        Engineering-Mechanical", "57        Engineering-Metallurgy & Material", "82        Economics", "0103000   Ethical Considerations", "0106000   Materials Research", "0108000   Software Development", "0306000   Energy Research & Resources", "0109000   Structural Technology", "54        Engineering-Civil", "72        Psychology Social Aspects", "0203000   Health", "0206000   Telecommunications", "0313040   Water Pollution", "0316000   Trace Contaminants", "0308000   Industrial Technology", "30        Computer Science & Engineering", "0201000   Agriculture", "45        Ecology", "0313000   Regional & Environmental", "0521800   Shelf & Esturine Ecosystem", "85        Political Sciences", "53        Engineering-Chemical", "16        Solid State Chem and Polymers", "0000912   Computer Science", "0510204   Data Banks & Software Design", "", "0400000   Industry University - Co-op", "0111000   Science Policy", "92        Science Policy", "86        Sociology", "71        Psychology Biological Aspects", "0107000   Operations Research", "0120000   Research Involving Recombinant DNA", "0204000   Oceanography", "0104000   Information Systems", "0510602   Ecosystem Dynamics", "51        Engineering-Aeronautical", "0510601   Biology of Pest Organisms", "0313020   Noise Pollution", "0512004   Analytical Procedures", "0312000   Population", "0311000   Polar Programs-Related", "0319000   Climate Related Activities", "52        Engineering-Astronautical", "0112000   System Theory", "0510701   Chemical Reaction Systems", "0521400   Plant Productivity", "14        Condensed Matter Physics", "0510403   Engineering & Computer Science", "84        Linguistics", "87        Law", "0302000   Biological Pest Control", "64        Environmental Biology", "44        Physical & Chemical Oceanography", "0304000   Disaster & Natural Hazard", "17        Materials Theory", "0208000   Water Resources", "0304010   Earthquake", "0510803   Mineral Leaching Process", "83        History", "0205000   Space", "0118000   Pollution Control", "0510100   Global Carbon Cycle", "0510301   Structure & Function", "0300000   Problem-Oriented", "0304030   Flood", "18        Materials NEC", "0309000   Land Use & Planning", "0510604   Analytic Tools", "0510205   International Economic Cooperation", "0510101   Climate Models", "15        Metals, Ceramics & Electronic Materials", "0314000   Rural Development", "0510401   Nerve Growth", "0510102   Role-Terrestrial Ecosystem", "0202000   Atmospheric Science-ICAS", "0313010   Air Pollution", "0113000   Animal Welfare", "0522100   High Technology Materials", "0116000   Human Subjects", "0512204   Nondestructive Measurement", "0207000   Transportation", "0521700   Marine Resources", "0317000   Urban Technology", "0114000   Endangered Species", "0110000   Technology Transfer", "63        Life Science Other Medical", "0510304   Electron & Energy Sources", "0510302   Energetics & Thermodynamc", "0510603   Synthetic Organic Chemistry", "0522400   Information Systems", "0304020   Extreme Wind", "0510802   Rock Properties-Mineral", "0100000   Special Interest", "0511603   Societal Aspects", "0510103   Physical Chemical Studies", "0318000   Weather Modifications", "0522300   Man-Machine Communication", "0510203   Model Validation", "0510804   Mineral Economics", "0102000   Data Banks", "0510200   Economic Instability", "0510704   Geophysical Monitoring", "0510300   Transformation of Nitrogen", "0510600   Integrated Pest Management", "65        Agricultural", "0313030   Solid Waste Disposal", "0512003   Collective Decision-Making", "0301000   Aging (Human)", "0303000   Conflict Resolution", "0511300   Nutrition", "0510402   Biomaterials-Short & Long Terms", "0510404   Sensry Physiology & Perc", "0601000   Facilities - Repair/Renovation", "70        Psychology", "0602000   Facilities - Replacement", "0512000   Population Redistribution", "0510801   Geodynamics & Mettallogen", "62        Life Science Clinical Medical", "0000904   Science Information", "0511601   Cloud & Precipitatn Process", "0510202   International Economic Modeling", "0315000   Social Data & Comm Development", "0304040   Fire", "0200000   Interagency", "0510400   Physically Disadvantaged", "0522500   Early Technology", "null", "0521500   Ozone & The Stratosphere", "0521200   Salt Tolerant Plants", "0520900   Substitute Materials", "0510201   Domestic Economic Modeling", "0512201   Improved Transient Measure", "0512202   Improved Resolution & Sen", "0510305   Photocatalysis", "0307000   Human Rights-Minority Rel", "0511301   Trace Element Content", "58        Engineering-Engineering Technology", "0512206   System Interaction", "0510303   Kinetics", "0520500   Computer Security & Privacy", "0304050   Accident", "0603000   Facilities - Combination", "0510800   Deep Mineral Deposits", "0521900   Mass Media & Socialization", "0119000   National Environmental Policy Act", "0115000   Historical Sites", "9900000     DATA NOT AVAILABLE", "0512205   Xray & Electron Beam Lith", "0510700   Deep Solution Mining", "0511303   Toxicology", "0510702   Reservoir Engineering", "0310000   Municipal Service", "0510703   Rock Fracture Mechanics", "0117000   Marine Mammal Protection", "0000908   Urban Problems", "0521000   Mineral Benefication", "0510104   Past Carbon Dioxide Level", "0305000   Drug Abuse", "0511600   Weather Modification Research", "0521100   Waste Resources", "0512200   Advanced Measurement Investment", "0511302   Role-Trace Element Component", "0305030   Drug Abuse Prevention", "0600000   Facilities", "0511602   Inadvertent Weather Effect", "0305010   Drug Abuse Effects Biol", "0512203   Improved Sensitivity-Minitr", "0121000   Proprietary & Privileged Information"]
newLabels = ["UNK", "UNK", "ENG", "MCS", "EAOS", "BAS", "PS", "UNK", "MCS", "PS", "SOC", "PS", "EAOS", "EAOS", "EAOS", "BAS", "SOC", "PS", "SOC", "UNK", "SOC", "ENG", "ENG", "UNK", "PS", "EAOS", "EAOS", "MCS", "PSY", "SOC", "BAS", "ENG", "ENG", "SOC", "SOC", "ENG", "MCS", "ET", "ET", "ENG", "PSY", "HLT", "MCS", "ET", "PS", "ET", "MCS", "BAS", "BAS", "EAOS", "EAOS", "SOC", "ENG", "PS", "MCS", "MCS", "UNK", "ET", "SOC", "SOC", "SOC", "PSY", "ENG", "BAS", "EAOS", "MCS", "EAOS", "ENG", "BAS", "SOC", "SOC", "SOC", "EAOS", "EAOS", "ENG", "SOC", "PS", "BAS", "PS", "MCS", "SOC", "SOC", "BAS", "BAS", "EAOS", "EAOS", "ENG", "EAOS", "EAOS", "EAOS", "SOC", "PS", "EAOS", "EAOS", "ET", "SOC", "EAOS", "ENG", "EAOS", "ET", "SOC", "EAOS", "EAOS", "SOC", "BAS", "EAOS", "EAOS", "EAOS", "EAOS", "ENG", "HLT", "ENG", "SOC", "EAOS", "SOC", "EAOS", "SOC", "BAS", "PS", "PS", "PS", "MCS", "EAOS", "EAOS", "UNK", "SOC", "PS", "EAOS", "SOC", "MCS", "SOC", "MCS", "SOC", "EAOS", "PS", "EAOS", "BAS", "EAOS", "SOC", "SOC", "SOC", "HLT", "ENG", "HLT", "ET", "PSY", "ET", "SOC", "EAOS", "HLT", "UNK", "EAOS", "SOC", "SOC", "PS", "SOC", "SOC", "SOC", "UNK", "EAOS", "BAS", "ENG", "SOC", "UNK", "UNK", "PS", "SOC", "PS", "ET", "SOC", "PS", "MCS", "SOC", "ET", "EAOS", "SOC", "EAOS", "SOC", "UNK", "PS", "EAOS", "BAS", "ET", "SOC", "EAOS", "EAOS", "SOC", "EAOS", "EAOS", "HLT", "EAOS", "ET", "ET", "PS", "HLT", "ET", "EAOS", "HLT", "UNK", "UNK"]
allLabels = {}

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
    global labels
    global newLabels
    global allLabels
    fieldDict = {}
    mainData = []

    parts = 3

    # Used to count the number of files
    counter = 0

    mainPath = "../../Data/NSF Dataset/Part "
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
                    # print(abstract.strip().replace("  "," ").replace("  "," "))
                    dataset = [abstract.strip().replace("  "," ").replace("  "," "),tags]
                    # dataset = [fixAbstract(abstract.strip()), tags]
                    mainData.append(dataset)
    

    # Print statistics
    print("Files found: " + str(counter))
    print("Fields found:")
    for i in fieldDict:
        print(i + "\t" + str(fieldDict[i]))
        
    badClassCount = 0
    # list of data
    for i in mainData:
        
        for j in range(len(i[1])):
            ind = labels.index(i[1][j])
            i[1][j] = newLabels[ind]
        # print(i)
        # print("\n-------------------------------------\n")
        i[1] = list(dict.fromkeys(i[1]))
        addClasses(i[1])

    sum = 0
    for i in classes:
        isBadClass = False
        if i in badClasses:
            isBadClass = True
        print(i + "\t" + str(classes[i]))
        # print("\"" + i + "\"")
        sum = sum + classes[i]
    # print("\nTotal = " + str(total))
    # print("Bad Labels = " + str(badClassCount))
    # print("% Bad Labels = " + str(round(badClassCount / total * 100, 2)) + "%")

    dataFiltered = [[i[0],[label for label in i[1] if label != "UNK"]] for i in mainData if ( "".join(i[1]) != "UNK" and len(i[1]) > 0 )]

    with open("../../Data/relabeledNSFfiltered.p","wb") as handle:
        pickle.dump(dataFiltered,handle)

    mainData = dataFiltered

    counter = 0

    for i in mainData:
        # If a specialization is not in allSpec keys, then add it
        for spec in i[1][0:1]:
            if(spec not in allLabels):
                allLabels[spec] = 0
            if(spec == "0000099   Other Applications NEC"):
                print(i)
            allLabels[spec] += 1

        counter+=1
            


    print("Number of Papers: " +str(len(mainData)))
    print(allLabels)

    
main()
>>>>>>> merge
