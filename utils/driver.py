from classifiers.classifier import Classifier
from utils.tweet_fetch import GetTweets

import tqdm

"""Class to orchestrate prediction generation"""
class Driver:

    """Configures model and metavariables like numTweets"""
    def __init__(self, model_name, api_key, api_secret, acc_tok, acc_secret, trainNew, daysBack_, numTweets_):
        trainDir = "./data/train/"
        self.tweetScraper = GetTweets(api_key, api_secret, acc_tok, acc_secret)
        self.classifier = Classifier(model_name, trainNew, trainDir)
        self.daysBack = daysBack_
        self.numTweets = numTweets_
    # end

    """Pulls numTweets tweets and runs prediction on them, then returns most common prediction"""
    def classify(self, ticker):
        tweets = self.tweetScraper.get_ticker_tweets(ticker, self.daysBack, self.numTweets)
        predictions = {"bearish" : 0, "bullish" : 0, "neutral" : 0}

        if len(tweets) == 0:
            return "invalid", 0

        print("Making predictions")
        for tweet in tqdm.tqdm(tweets):
            pred = self.classifier.classify(tweet)
            predictions[pred] += 1
        
        maxLabel = ""
        maxPerc = 0
        for label, num in predictions.items():
            if (num/self.numTweets) > maxPerc:
                maxLabel = label
                maxPerc = num/self.numTweets
        

        return maxLabel, maxPerc
