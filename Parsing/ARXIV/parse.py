from os import listdir
from os.path import isfile, join
import xml.etree.ElementTree as ET
import pickle

#path to dataset, I wouldn't change it since the gitignore might include it 
datasetPath = "./dataset"

#Get all files in the dataset path
onlyFiles = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f))]

#Different dict for grouping papers differently
bySpecialization = {}
byAllCategory = {}
byFirstCategory = {}

for name in onlyFiles:
    
    file = open(datasetPath+"/"+name,mode="r",encoding="UTF-8")
    
    tree = ET.parse(file)

    file.close()
    
    # Information for a paper is listed under a <record> tag in the dataset
    # Format of the XML for a record for a paper:
    # <record>
    #   <metaPrefix metadata> <-- Metaprefix is a link attached to the metadata tag, required in all children of metadata
    #       <arXiv> 
    #           <header>
    #               <setSpec> Paper Specialization Here </setSpec>
    #               <categories> Paper Research Categories Here </categories>
    #               <abstract> Paper abstract Here </abstract>
    #       </arXiv>
    #   </metadata>
    # </record>

    # Get all record tags for the file, therefore getting tags for all paper records
    allPapers = tree.getroot().find("ListRecords").findall("record")
    
    for paper in allPapers:
        
        try:
            
            # get all the fields we will need for classification out of the record
            metaPrefix = "{http://arxiv.org/OAI/arXiv/}"
            info = paper.find("metadata").find(metaPrefix+"arXiv")
            specialization = paper.find("header").find("setSpec").text.split(":")[0]
            categories = info.find(metaPrefix+"categories").text.split(" ")
            abstract = info.find(metaPrefix+"abstract").text
            
            # group into specialization, first category, and all categories

            if(not specialization in bySpecialization):
                bySpecialization[specialization] = []
            
            bySpecialization[specialization].append(abstract)
            
            if(not categories[0] in byFirstCategory):
                byFirstCategory[categories[0]] = []
            
            byFirstCategory[categories[0]].append(abstract)
            
            for category in categories:
            
                if(not category in byAllCategory):
                    byAllCategory[category] = []
            
                byAllCategory[category].append(abstract)
        
        # This would occur if the OAI harvest was interrupted while collecting data
        except:
            print("Record " +str(paper.find("header").find("identifier").text) + " in " +str(name) + " does not contain required info")

with open("arXivFirstCategory", 'wb') as handle:
    pickle.dump(byFirstCategory, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("arXivAllCategory", 'wb') as handle:
    pickle.dump(byAllCategory, handle, protocol=pickle.HIGHEST_PROTOCOL)
with open("arXivSpecialization", 'wb') as handle:
    pickle.dump(bySpecialization, handle, protocol=pickle.HIGHEST_PROTOCOL)


