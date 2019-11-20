from os import listdir
from os.path import isfile, join
from multiprocessing import Pool, Process
import xml.etree.ElementTree as ET
import pickle

# path to dataset, I wouldn't change it since the gitignore might include it 
datasetPath = "./dataset"

# Get all files in the dataset path

onlyFiles = [f for f in listdir(datasetPath) if isfile(join(datasetPath, f))]


# worker function to parse a dataset file given the file name 

def parseFile(name):

    global datasetPath

    # Output array of the parsed data

    out = []

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
    #           </header>
    #       </arXiv>
    #   </metadata>
    # </record>

    # Get all record tags for the file, therefore getting tags for all paper records

    allPapers = tree.getroot().find("ListRecords").findall("record")
    
    for paper in allPapers:
        
        metaPrefix = "{http://arxiv.org/OAI/arXiv/}"
        info = 0
        specialization = 0
        categories = 0
        abstract = 0

        try:
            
            # get all the fields we will need for classification out of the record

            metaPrefix = "{http://arxiv.org/OAI/arXiv/}"
            info = paper.find("metadata").find(metaPrefix+"arXiv")
            specialization = paper.find("header").find("setSpec").text.split(":")[0]
            categories = info.find(metaPrefix+"categories").text.split(" ")
            abstract = info.find(metaPrefix+"abstract").text
        
            for category in categories:
                
                out.append([category,abstract])
        
        # This would occur if the OAI harvest was interrupted while collecting data

        except:
            
            print("Record " +str(paper.find("header").find("identifier").text) + " in " +str(name) + " does not contain required info")

    return out

# Pool of processes for parsing the files 
fileParsingPool = Pool()

# For each filename in only files, pass that into parseFile and run as a separate process in the fileParsingPool, allowing parallelism

map = fileParsingPool.map_async(parseFile,onlyFiles)

# Close off the parsing process pool and then join it to the main process so we wait until all results retrieved

fileParsingPool.close()

fileParsingPool.join()

number = 0

result = []

# Concatenate each file's parsed results onto the results array, this one can take a while since it's single thread

for res in map.get(timeout=0):

    result = result + res
    

with open("arXivAll.p", 'wb') as handle:
    
    pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)
