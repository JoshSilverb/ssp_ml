import tweepy
import datetime

class GetTweets:
    def __init__(self, consumer_key, consumer_secret, access_token, access_token_secret):
        self.auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
        self.auth.set_access_token(access_token, access_token_secret)
        self.api = tweepy.API(self.auth)
    # end


    def get_ticker_tweets(self, ticker, daysBack, numTweets):
        #tweetTexts = []
        now = datetime.datetime.now()
        d = datetime.timedelta(days = daysBack)

        date = now - d

        # Define the search term and the date_since date as variables
        ticker = "$" + ticker
        date_since = date.strftime("%Y-%m-%d")
        print("Collecting tweets for", ticker, "from", date_since)

        # Collect tweets
        tweets = tweepy.Cursor(self.api.search,
                    q=ticker,
                    lang="en",
                    since=date_since).items(numTweets)

        tweets = [tweet.text for tweet in tweets]

        print("Found {} tweets".format(len(tweets)))
        
        return tweets
    # end