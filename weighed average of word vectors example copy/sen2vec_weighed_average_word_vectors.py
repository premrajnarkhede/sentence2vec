from __future__ import division
import spacy
import requests
import json
from scipy import spatial
import random
from sklearn.metrics import roc_curve, auc, log_loss
from unidecode import unidecode
from sklearn.linear_model import LogisticRegression
import numpy as np
import csv
import math
import re

class readData(object):
    def __init__(self):
        pass
    def getRecentTweets(self,load_from_file=False):
        if load_from_file==False:
            url = '' #I removed url as it was pointing to private unprotected resource
            data = requests.get(url).text
            tweet_data = open("tweets.json","wb")
            tweet_data.write(data)
            tweet_data.close()
        data = open("tweets.json","rb").read()
        dicta = json.loads(data)
        tweets = []
        for doc in dicta["response"]["docs"]:
            tweet_text = doc.get("tweet_text","")
            tweets.append(tweet_text)
        return tweets

def removeAllPunctuations(g):
    g= g.replace("."," ")
    g= g.replace(","," ")
    g= g.replace("'","")
    g= g.replace("-"," ")
    g= g.replace("/"," ")
    g= g.replace(":"," ")
    g= g.replace(";"," ")
    g= g.replace('"',"")
    g= g.replace("*","")
    g= g.replace("?"," ")
    g= g.replace("&","and")
    g= g.replace("+"," ")
    g= g.replace("["," ")
    g= g.replace("]"," ")
    g= g.replace("("," ")
    g= g.replace(")"," ")
    g= g.replace("<"," ")
    g= g.replace(">"," ")
    g= g.replace("="," ")
    g= g.replace(","," ")
    g= re.sub( '\s+', ' ', g ).strip()
    return g

def getLog(frequency):
    return 1/math.log(frequency)

class weighedVectorSimilarity(object):
    def __init__(self):
        self.readWeights()
        reader = readData()
        self.tweets = reader.getRecentTweets(load_from_file=True)
        self.nlp = spacy.load('en')
    def readWeights(self):
        print "Reading weights"
        data = open("words-frequency.txt","rb").read()
        data = data.split("\n")
        self.word_weight = {}
        for d in data:
            wordpair = d.split(" ")
            self.word_weight[wordpair[0]] = int(wordpair[1])
        print "Done reading weights"
    def getAverageWeighedVectorForTweet(self,tweet):
        tweet = removeAllPunctuations(tweet.lower())
        tweet_doc = self.nlp(tweet)
        vectors = []
        weights = []
        for token in tweet_doc:
            if token.has_vector:
                if len(token.text)<3:
                    continue
                frequency = self.word_weight.get(token.text.lower(),10)
                weight = getLog(frequency)
                vectors.append(token.vector)
                weights.append(weight)
                #print token.text,weight,frequency
        #print weights
        try:
            doc_vector = np.average(vectors,weights=weights,axis=0)
        except:
            print "Exception"
            return tweet_doc.vector
        return doc_vector
    def getSentenceVectors(self):
        vectors = []
        for tweetindex,tweet in enumerate(self.tweets):
            if tweetindex%100==0:
                print tweetindex
            vector = self.getAverageWeighedVectorForTweet(tweet)
            vectors.append(vector)
        self.tree = spatial.KDTree(vectors)
    def findNearestVector(self):
        while 1:
            text = unicode(raw_input("Enter text:"))
            vector = self.getAverageWeighedVectorForTweet(text)
            results= self.tree.query(vector,10)
            for rindex,result in enumerate(results[1]):
                print "Distance: ",results[0][rindex]," Tweet Text: ",self.tweets[result]
                print "\n"
    def QuoraEvaluation(self):
        data = open("../quora_duplicate_questions.tsv","rb").read()
        data = data.split("\n")
        self.dataset = []
        for lineindex,line in enumerate(data):
            if lineindex==0:
                continue
            if lineindex%100==0:
                print lineindex
            if lineindex ==500000:
                break
            lineitems = line.split("\t")
            try:
                question1 = unicode(unidecode(lineitems[3]))
                question2 = unicode(unidecode(lineitems[4]))
                evaluation = int(lineitems[5])
            except IndexError,e:
                print e
                continue
            self.dataset.append([question1,question2,evaluation])
        random.shuffle(self.dataset)
        split_point = int(0.8*len(self.dataset))
        self.train,self.test = self.dataset[:split_point],self.dataset[split_point:]
        
        
        ##Training
        predictions = []
        evaluations = []
        for entryindex,entry in enumerate(self.train):
            
            question1 = entry[0]
            question2 = entry[1]
            evaluation = entry[2]
            vector1 = self.getAverageWeighedVectorForTweet(question1)
            vector2 = self.getAverageWeighedVectorForTweet(question2)
            distance = spatial.distance.cosine(vector1,vector2)
            if entryindex%100==0:
                print entryindex,question1,question2,evaluation,distance
            if not np.isnan(distance):
                predictions.append([distance])
                evaluations.append([evaluation])
        predictions = np.array(predictions)
        evaluations = np.array(evaluations)
        regr = LogisticRegression()
        regr.fit(predictions,evaluations)
        print "Accuracy on training set",regr.score(predictions,evaluations)
        
        predictions_test = []
        evaluations_test = []
        sample_output_questions = []
        for entryindex,entry in enumerate(self.test):
            
            question1 = entry[0]
            question2 = entry[1]
            evaluation = entry[2]
            vector1 = self.getAverageWeighedVectorForTweet(question1)
            vector2 = self.getAverageWeighedVectorForTweet(question2)
            distance = spatial.distance.cosine(vector1,vector2)
            if entryindex%100==0:
                print entryindex,question1,question2,evaluation,distance
            
            if not np.isnan(distance):
                if entryindex < 1000:
                    sample_output_questions.append([question1,question2])
                predictions_test.append([distance])
                evaluations_test.append([evaluation])
        print "Accuracy on testing set",regr.score(predictions_test,evaluations_test)
        my_file=open("sample_output.csv","wb")
        file_output = csv.writer(my_file, delimiter=',',quotechar='"',lineterminator='\n')
        for entryindex,sample_entry in enumerate(sample_output_questions):
            output = regr.predict(np.array(predictions_test[entryindex]).reshape(1,-1))
            file_output.writerow([sample_entry[0],sample_entry[1],output,predictions_test[entryindex][0],evaluations_test[entryindex]]) 
        
if __name__ == '__main__':
    avs = weighedVectorSimilarity()
    avs.QuoraEvaluation()
    # avs.getSentenceVectors()
    # avs.findNearestVector()