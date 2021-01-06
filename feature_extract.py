import re
import string

import textacy
import spacy
from spacy.lemmatizer import Lemmatizer
from spacy.lookups import Lookups
import en_core_web_sm


class FeatureExtractor:
    def __init__(self):
        self.featureVector = []
        self.features = {}
        self.nlp = en_core_web_sm.load()
    #end

    # cleans up tweet by removing characters and urls with regex and returns stripped 
    def clean_tweet(self, tweet):
        tweet = tweet.lower()

        # erase links starting with https
        tweet = re.sub(r'^https?:\/\/.*[\r\n]*', '', tweet)
        # erase links starting with www
        tweet = re.sub(r'^www?:\/\/.*[\r\n]*', '', tweet)

        #Remove @username 
        tweet = re.sub(r'@[^\s]+',' ',tweet)
        #Remove $companytag 
        tweet = re.sub(r'\$[^\s]+',' ',tweet)
        #Replace #word with word
        tweet = re.sub(r'#([^\s]+)', r'\1', tweet)
        #remove punctuation
        tweet = tweet.replace('\'','')
        tweet = re.sub(r'[%s]' % re.escape(string.punctuation), ' ', tweet)
        #Remove additional white spaces
        tweet = re.sub(r'[\s]+', ' ', tweet)
        return tweet
    # end


    def get_feature_vector(self, tweet):
        # remove extra chars and stuff
        tweet = self.clean_tweet(tweet)
        doc = self.nlp(tweet)

        bigrams = [token for token in doc if not token.is_stop]
        bigrams = [token for token in bigrams if len(token) > 1]

        bigrams = list(textacy.extract.ngrams(bigrams, 2, filter_stops=False))

        processed_bigrams = []

        for bigram in bigrams:
            processed_bigrams.append([bigram[0].lemma_, bigram[1].lemma_])

        return processed_bigrams
    # end


        

