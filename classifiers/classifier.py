
from classifiers import NAME_TO_MODEL_MAP
import classifiers
from keras.models import load_model
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
import numpy as np
from utils.feature_extract import FeatureExtractor
import pickle
import json
import tqdm

from classifiers import VALID_MODEL_NAMES



"""Abstraction for all other classifiers"""
class Classifier:
    def __init__(self, model_name, should_train_new, train_data_dir) -> None:
        if not model_name in VALID_MODEL_NAMES:
            raise NameError(f"Invalid model name - must be one of {VALID_MODEL_NAMES}")

        self.model_name = model_name
        self.labels = ["bearish", "bullish", "neutral"]
        self.trainDir = train_data_dir
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.tokenizer = Tokenizer(num_words=5000)
        self.f = FeatureExtractor()
        self.config = {}
        model_path = f"./data/models/{model_name}.h5"

        if not should_train_new and exists(model_path) and exists("./data/pickles/tokenizer.pickle") and exists("./data/config.json"):
            self.model = load_model(model_path)
            with open("./data/pickles/tokenizer.pickle", 'rb') as tokFile:
                self.tokenizer = pickle.load(tokFile)
            with open("./data/config.json", 'r') as jsonFile:
                self.config = json.load(jsonFile)
            
        else:
            if not exists(model_path):
                print("Model save file not found - retraining model...")
                self.load_data()
                self.train_model()
            elif should_train_new:
                print("Should_train_new = True - retraining model...")
                self.load_data()
                self.train_model()
            elif not exists("./data/pickles/tokenizer.pickle") or not exists("./data/config.json"):
                print("Tokenizer or config not found - reloading tweet corpus from train data directory")
                self.load_data()
            
    # init

    def load_data(self):
        
        print("Loading data...")
        # create pandas dataframe with texts and labels for each labelled file
        dict_list = []
        for label in self.labels:
            file = open(self.trainDir + label + ".txt", 'r')
            for line in file:
                tweet = self.f.clean_tweet(line)
                dict_list.append({"text" : tweet, "label" : label})
        df = pd.DataFrame(dict_list)
        
        # separete dataframe into tweets and labels and create train/test split
        tweets = df['text'].values
        y = df['label'].values
        tweets_train, tweets_test, self.y_train, self.y_test = train_test_split(tweets, y, test_size=0.20, random_state=1000)
        print(tweets_train.shape[0], "training tweets,", tweets_test.shape[0], "testing tweets")

        # convert categorical labels into one-hot representation
        enc = OneHotEncoder()
        self.y_train = self.y_train.reshape(-1,1)
        self.y_train = enc.fit_transform(self.y_train).toarray()
        self.y_test = self.y_test.reshape(-1,1)
        self.y_test = enc.fit_transform(self.y_test).toarray()

        # tokenize tweets in prep to sent into model
        
        self.tokenizer.fit_on_texts(tweets_train)

        # save tokenizer for later runs where model isn't retrained
        with open("./data/pickles/tokenizer.pickle", 'wb') as tokFile:
            pickle.dump(self.tokenizer, tokFile)

        self.X_train = self.tokenizer.texts_to_sequences(tweets_train)
        self.X_test = self.tokenizer.texts_to_sequences(tweets_test)

        vocab_size = len(self.tokenizer.word_index) + 1      # add 1 because 0 is reserved and has no associated word

        # find longest array length to pad all arrays to below
        max_len = 0
        for arr in self.X_train:
            if len(arr) > max_len:
                max_len = len(arr)

        # pad text_to_sequences arrays with 0s
        self.X_train = pad_sequences(self.X_train, padding='post', maxlen=max_len)
        self.X_test = pad_sequences(self.X_test, padding='post', maxlen=max_len)


        self.config["max_len"] = max_len
        self.config["vocab_size"] = vocab_size

        with open("./data/config.json", 'w') as jsonFile:
            json.dump(self.config, jsonFile)

        print("Done")

        return self.X_train, self.X_test, self.y_train, self.y_test
    
    # load_data

    
    def train_model(self):
        self.model = NAME_TO_MODEL_MAP[self.model_name].train_model(
                                                            self.config["vocab_size"],
                                                            self.config["max_len"],
                                                            self.X_train,
                                                            self.y_train,
                                                            self.X_test,
                                                            self.y_test)
        
    
    def classify(self, tweetText):
        if self.model_name == "svc":
            f = FeatureExtractor()
            tweetText = str(tweetText)
            featureVector = f.get_feature_vector(tweetText)
            features = dict([(tuple(word), True) for word in featureVector])

            prediction = self.model.classify(features)

            return prediction

        # else:
        tweetText = self.f.clean_tweet(tweetText)
        vector = self.tokenizer.texts_to_sequences([tweetText])
        vector = pad_sequences(vector, padding='post', maxlen=self.config["max_len"])
        [prediction] = self.model.predict(vector)
        
        index = int(np.argmax(prediction, axis=0))

        return self.labels[index]