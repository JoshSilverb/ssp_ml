from classifier import Classifier
from tweet_fetch import GetTweets

import sys
import os
import tqdm
import argparse

class Driver:
    def __init__(self, api_key, api_secret, acc_tok, acc_secret, trainNew, daysBack_, numTweets_):
        trainDir = "./data/train/"
        self.tweetScraper = GetTweets(api_key, api_secret, acc_tok, acc_secret)
        self.classifier = Classifier(trainNew, trainDir)
        self.daysBack = daysBack_
        self.numTweets = numTweets_
    # end

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

            
def parse_args():
    parser: argparse.ArgumentParser = argparse.ArgumentParser()
    parser.add_argument('-t',
                        action='store', dest='train',
                        type=str,
                        required=True,
                        help='train new model [True/False]')
    parser.add_argument('-n',
                        action='store', dest='num',
                        type=int,
                        required=True,
                        help='number of tweets to look for')
    parser.add_argument('-d',
                        action='store', dest='daysBack',
                        type=int,
                        required=True,
                        help='maximum number of days before today to search through')

    return parser.parse_args(sys.argv[1:])

if __name__ == "__main__":
    
    '''
    API_key = input("API key: ")
    API_secret = input("API key secret: ")
    access_token = input("Access token: ")
    access_secret = input("Access token secret: ")
    
    system('clear')
    '''

    API_key = "UjZiYFkC1UJFHDVUHdCikJQYG"
    API_secret = "PFvAZfR51zNVK03ipLWpyBOjJCKjibRx1YvSR9Am5LplyIWunt"
    access_token = "2911122124-XpSTbAnFvYmIuSPaZMvXVvFrmCvMNNSxGXs3sfU"
    access_secret = "uHdAiCoQPkV6Ng9vv0sHXadTDF19PdCYjhFKOsWkQw0V7"
    
    
        
    if len(sys.argv) > 1:
        config = parse_args()
        if config.train.lower() in ('no', 'false', 'f', 'n', '0'):
            train = False
        elif config.train.lower() in ('yes', 'true', 't', 'y', '1'):
            train = True
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

        
        driver = Driver(API_key, API_secret, access_token, access_secret, train, config.daysBack, config.num)
    else:
        # default setup
        driver = Driver(API_key, API_secret, access_token, access_secret, False, 2, 50)
    
    col, l = os.get_terminal_size()
    print('='*col)
    print("Stock Sentiment Predictor - Josh Silverberg - https://github.com/JoshSilverb/ssp_ml")
    print("Type \"help\" for help")

    ticker = input(">> Stock ticker name: ")
    while len(ticker) > 0:
        if ticker == "help":
            print("\t================ help ================")
            print("\t>> Stock ticker name: [ticker name for stock to investigate]")
            #print("\t>> Max days back: []")
            print("\t======================================")
            ticker = input(">> Stock ticker name: ")
            continue
        #daysBack = int(input(">> Max days back: "))

        label, percent = driver.classify(ticker)
        
        if label == "invalid":
            print("No tweets found - likely invalid ticker")
        else:
            print("Predicted:", label, "with", str(int(percent*100)) + "% confidence")


        ticker = input(">> Stock ticker name: ")


