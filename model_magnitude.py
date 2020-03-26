import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
import math
from scipy import spatial
import numpy as np

def termFrequency(term, doc):       
    # Splitting the document into individual terms 
    normalizeTermFreq = doc.lower().split()  
  
    # Number of times the term occurs in the document 
    term_in_document = normalizeTermFreq.count(term.lower())  
  
    #normalize value
    if term_in_document > 0:
        term_in_document = 1 + math.log(term_in_document)  
  
    return term_in_document 

def inverseDocumentFrequency(term, allDocs): 
    num_docs_with_given_term = 0

    # Iterate through all the documents 
    for doc in allDocs: 
        if term.lower() in doc.lower().split(): 
            num_docs_with_given_term += 1
  
    if num_docs_with_given_term > 0: 
        # Total number of documents 
        total_num_docs = len(allDocs)  
  
        # Calculating the IDF  
        idf_val = math.log(float(total_num_docs) / num_docs_with_given_term) 
        return idf_val 
    else: 
        return 0

#read data into a dataframe
data = pd.read_csv('data.txt', sep=",", header=None)
data.columns = ["transcript", "url"]

#get a list of transcipts to calculate the idf values
documents = data["transcript"].tolist()

phrase = input("Please enter a query to search: ")
phraseList = phrase.split()
tmpList = [None] * len(phraseList)
phrase_idfs = [None] * len(phraseList)
for i in range(len(phraseList)):
    phrase_tf = termFrequency(phraseList[i], phrase)
    phrase_idf = inverseDocumentFrequency(phraseList[i], documents)
    #since the idf is how many documents this word appears in, just calculate it once and put it in a list!
    phrase_idfs[i] = phrase_idf
    tmpList[i]  = phrase_tf * phrase_idf

phrase_tfidf = tmpList

data = data.values.tolist()

class Document():
	def __init__(self, link, magnitude):
		self.link = link
		self.magnitude = magnitude

documents = []

for array in data:
    document = array[0]
    link = array[1]
    tmpList = [None] * len(phraseList)
    for i in range(len(phraseList)):
        document_tf = termFrequency(phraseList[i], document)
        #print("the phrase {} has a frequency of {}".format(phraseList[i], document_tf))
        document_idf = phrase_idfs[i]
        tmpList[i]  = document_tf * document_idf
    document_tfidf = tmpList
    magnitude = np.linalg.norm(document_tfidf)
    documents.append(Document(link, magnitude))
    

#sort documents by their magnitudes
#One line sort function method using an inline lambda function lambda x: x.date
documents.sort(key=lambda x: x.magnitude, reverse=True)

#print out top 10 most relavent talks, if possible
for i in range(10):
    if documents[i].magnitude != 0.0:
        print("The talk {} had a magnitude of {}".format(documents[i].link, documents[i].magnitude))
    else:
        break