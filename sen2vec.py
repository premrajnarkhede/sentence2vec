import spacy
import requests
import json
from scipy import spatial

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

class averageVectorSimilarity(object):
    def __init__(self):
        reader = readData()
        self.tweets = reader.getRecentTweets(load_from_file=True)
        self.nlp = spacy.load('en')
        
    def getAverageVectorForTweet(self,tweet):
        tweet_doc = self.nlp(tweet)
        return tweet_doc.vector
    def getSentenceVectors(self):
        vectors = []
        for tweetindex,tweet in enumerate(self.tweets):
            if tweetindex%100==0:
                print tweetindex
            vector = self.getAverageVectorForTweet(tweet)
            vectors.append(vector)
        self.tree = spatial.KDTree(vectors)
    def findNearestVector(self):
        while 1:
            text = unicode(raw_input("Enter text:"))
            vector = self.getAverageVectorForTweet(text)
            results= self.tree.query(vector,10)
            print results
            for rindex,result in enumerate(results[1]):
                print "Distance: ",results[0]," Tweet Text: ",self.tweets[result],
            
if __name__ == '__main__':
    avs = averageVectorSimilarity()
    avs.getSentenceVectors()
    avs.findNearestVector()