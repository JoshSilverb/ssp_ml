from keras.models import load_model
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
import numpy as np
from feature_extract import FeatureExtractor
import pickle
import json
import tqdm

class ConvClassifier:
    def __init__(self, trainNew, trainDirectory):
        self.labels = ["bearish", "bullish", "neutral"]
        self.trainDir = trainDirectory
        self.X_train = []
        self.X_test = []
        self.y_train = []
        self.y_test = []
        self.tokenizer = Tokenizer(num_words=5000)
        self.f = FeatureExtractor()
        self.config = {}


        if not trainNew and exists("./data/models/cnn.h5") and exists("./data/pickles/tokenizer.pickle") and exists("./data/config.json"):
            self.model = load_model('./data/models/cnn.h5')
            with open("./data/pickles/tokenizer.pickle", 'rb') as tokFile:
                self.tokenizer = pickle.load(tokFile)
            with open("./data/config.json", 'r') as jsonFile:
                self.config = json.load(jsonFile)
            
        else:
            if not exists("./data/models/cnn.h5"):
                print("Model save file not found - training model")
                self.load_data()
                self.train_model()
            elif not exists("./data/pickles/tokenizer.pickle") or not exists("./data/config.json"):
                print("Tokenizer or config not found - reloading tweet corpus from train data directory")
                self.load_data()
            
    # end
    



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


        # save config data into config 
        self.config["max_len"] = max_len
        self.config["vocab_size"] = vocab_size

        with open("./data/config.json", 'w') as jsonFile:
            json.dump(self.config, jsonFile)

        print("Done")

        return self.X_train, self.X_test, self.y_train, self.y_test
    # end
        



    def train_model(self):
        if len(self.X_train) == 0:
            self.load_data()

        print("Training model...")

        self.model = Sequential()
        self.model.add(layers.Embedding(self.config["vocab_size"], 50, input_length=self.config["max_len"]))
        self.model.add(layers.Conv1D(128, 5, activation='relu'))
        self.model.add(layers.GlobalMaxPool1D())
        self.model.add(layers.Dense(10, activation='relu')) # 4340*10= 43400 weights + 10 bias = 43410 params
        self.model.add(layers.Dense(3, activation='softmax'))

        self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        self.model.summary()

        self.model.fit(self.X_train, self.y_train, epochs=100, verbose=False, validation_data=(self.X_test, self.y_test), batch_size=15)

        self.model.save("data/models/cnn.h5")
        print("Done")
        return self.model
    # end


    def classify(self, tweetText):
        tweetText = self.f.clean_tweet(tweetText)
        vector = self.tokenizer.texts_to_sequences([tweetText])
        vector = pad_sequences(vector, padding='post', maxlen=self.config["max_len"])
        [prediction] = self.model.predict(vector)
        
        index = int(np.argmax(prediction, axis=0))

        return self.labels[index]



if __name__ == "__main__":
    cfr = ConvClassifier(False, "./data/train/")

    for label in cfr.labels:
        print("Classifying",label,"tweets")
        with open("data/train/"+label+".txt", 'r') as file:
            correct = 0
            count = 0
            for line in tqdm.tqdm(file.readlines()):
                pred, perc = cfr.classify(line)
                if pred == label:
                    correct += 1
                count += 1
            print("correct:", correct, "count:", count, str(float(correct/count))+"%")
        print("done")
    