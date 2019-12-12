import pickle
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk import ngrams
from collections import OrderedDict
from multiprocessing import Pool,Manager
import multiprocessing as mp
import gc

filename = "relabeledNSFfiltered.p"

# Top N entries to be found

topN = 500

# ngram length

n = 1

# if your data has already been tokenized and split into ngrams with n matching that above

pretokened = False

def countSentenceNGram(sentences):

	global n

	v_dict = {}

	a_counter = 0

	gram_counter = 0

	for sentence in sentences[1]:

		if(not pretokened):

			sentence = list(word_tokenize(sentence))

		else:

			sentence = sentence.split(" ") 
		
		for i in range( ( len(sentence)-n+1 ) ):
			
			word = " ".join( sentence[i:i+n] )
			
			if word not in v_dict:
			
				v_dict[word] = 0

			v_dict[word] += 1
			gram_counter += 1

		if(a_counter % 10000 == 0):

			print(sentences[0],a_counter,"/",len(sentences[1]))

		a_counter += 1

	sortedList = sorted(v_dict.keys() , key = lambda x: v_dict[x], reverse=True)

	return (sentences[0],v_dict,sortedList,gram_counter)

# Group abstracts into a dictionary where highest level label is the specialization and the lower is abstracts in that spec

def groupByLabelIntoDict(f):

	labels = OrderedDict()

	data = pickle.load(open(f, "rb"))

	for a in data:

		for l in a[1][:1]:

			if l not in labels:

				labels[l] = []

			labels[l].append(a[0])

	del data

	data = None

	return labels

def topNotIn(i,ngramRes):

	global topN

	counter = 0

	c = 0

	topNList = []

	this_label = ngramRes[i][0]

	c_grams_this_label = ngramRes[i][3]

	while c < len(ngramRes[i][2]) and counter < topN:

		ngram = ngramRes[i][2][c]

		c_ngram_this_label = ngramRes[i][1][ngram]

		p_ngram_this_label = c_ngram_this_label / c_grams_this_label

		isNotIn = True

		for i2 in ngramRes:

			other_label = i2[0]

			if(ngram in i2[1] and this_label != other_label):

				c_ngram_other_label = i2[1][ngram]

				c_ngrams_other_label = i2[3]

				p_ngram_other_label = c_ngram_other_label / c_ngrams_other_label

				# The likelyhood that the ngram is found for a abstract of this ngram is lower than one of a different label

				if( p_ngram_this_label <= p_ngram_other_label * 3):

					isNotIn = False

					# break loop to avoid any more comparisons 

					break  

		if( isNotIn ): 

			topNList.append( ( ngram,c_ngram_this_label ) )

			counter += 1

			# if(counter%(topN/10) == 0):

			# 	print(ngramRes[i][0],counter)

		c += 1

	print(this_label,"Completed")

	# return the label that the list is associated with and the list itself 

	return ( this_label , topNList )

# async functions for pool with apply_async need a callback for some reason
# callback function for getting topN lists 

topNRes = []

def topNCallback(result):

	global topNRes

	topNRes.append(result)

if __name__ == '__main__':

	# for each new process created, spawn a new process rather than fork since windows does not support fork
	# this has the implication that child processes spawned by this one do not share memory from the parent in the same way they do with fork
	# if we were only developing on unix based systems that support fork we could have left this to default and behavior would be the same across systems
	# learn more about start methods here https://docs.python.org/3.4/library/multiprocessing.html

	mp.set_start_method("spawn")

	# create ordered dictionary of the abstracts with key being the abstract label and value being the abstracts for the label

	groupedDict = groupByLabelIntoDict(filename)

	# create list of tuples of ( key, value ) from the grouped dict to pass into the functions 

	tuples = []

	for k in groupedDict:

		tuples.append((k,groupedDict[k]))

	# sort the tuples by number of abstracts associated with each to speed up processing when one label has many more

	tuples.sort(key=lambda tup: len(tup[1]),reverse=True)

	del groupedDict

	groupedDict = None 

	# force memory collection

	gc.collect() 

	# pool of processes to get our ngram counts for all abstracts in each label

	ngramPool = mp.Pool()

	map = ngramPool.map_async(countSentenceNGram,tuples)

	ngramPool.close()

	ngramPool.join()

	ngramRes = map.get(timeout=0)

	del tuples

	tuples = None

	print("Got ngram results")

	# ngramRes = manager.list(list(ngramRes))

	ngramRes = list(ngramRes)

	topNPool = mp.Pool()

	map = []
	
	for i in range( len(ngramRes) ):

		map.append( topNPool.apply_async(topNotIn, args = (i,ngramRes), callback = topNCallback ) )

	topNPool.close()

	topNPool.join()

	for i in map:

		i.wait()

	del ngramRes

	ngramRes = None

	topSeqString = ""

	topSeqArr = []

	for top in topNRes:

		print("----------- TOP",topN,"FOR",top[0],"-----------------")

		for i in range(len(top[1])):

			rank = i

			ngram = top[1][i][0]

			c_ngram = top[1][i][1]

			print(i,ngram,c_ngram)

			topSeqArr.append(ngram)

			topSeqString += "\"" + ngram +"\"" + ", "

		print("----------- TOP" , topN , "FOR" , top[0] , "-----------------")

	with open("top"+str(topN)+filename.split(".")[0]+str(n)+"grams.p","wb") as handle:

		pickle.dump(topSeqArr,handle)

	print(topSeqString.replace("\\","\\\\"))

# exit()