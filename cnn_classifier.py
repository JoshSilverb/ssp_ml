from keras.models import load_model
from keras.models import Sequential
from keras import layers
from keras.preprocessing.text import Tokenizer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from os.path import exists
import pandas as pd
import numpy as np
from feature_extract import FeatureExtractor

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


        if not trainNew and exists("./data/models/cnn.h5"):
            self.model = load_model('./data/models/cnn.h5')
            
        else:
            print("Training model")
            self.load_data()
            self.train_model()
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
        #print(tweets_train.shape[0], "training tweets,", tweets_test.shape[0], "testing tweets")
        return tweets_train, tweets_test, self.y_train, self.y_test


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

        self.X_train = self.tokenizer.texts_to_sequences(tweets_train)
        self.X_test = self.tokenizer.texts_to_sequences(tweets_test)

        self.vocab_size = len(self.tokenizer.word_index) + 1      # add 1 because 0 is reserved and has no associated word

        # find longest array length to pad all arrays to below
        max_len = 0
        for arr in self.X_train:
            if len(arr) > max_len:
                max_len = len(arr)
        print(max_len)

        # pad text_to_sequences arrays with 0s
        from keras.preprocessing.sequence import pad_sequences

        self.X_train = pad_sequences(self.X_train, padding='post', maxlen=max_len)
        self.X_test = pad_sequences(self.X_test, padding='post', maxlen=max_len)
        self.max_len = max_len

        print("Done")

        return self.X_train, self.X_test, self.y_train, self.y_test
    # end
        



    def train_model(self):
        if len(self.X_train) == 0:
            self.load_data()

        print("Training model...")

        self.model = Sequential()
        self.model.add(layers.Embedding(self.vocab_size, 50, input_length=self.max_len))
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
        vector = self.tokenizer.texts_to_sequences([tweetText])
        print(vector)
        prediction = self.model.predict([vector])
        
        index = np.argmax(prediction, axis=1)

        return self.labels[index], prediction[index]



if __name__ == "__main__":
    cfr = ConvClassifier(False, "./data/train/")

    for label in cfr.labels:
        with open("data/train/"+label+".txt", 'r') as file:
            correct = 0
            count = 0
            for line in file.readlines():
                line = cfr.f.clean_tweet(line)
                pred, perc = cfr.classify(line)
                if pred == label:
                    correct += 1
                count += 1
            print("correct:", correct, "count:", count, str(int(correct/count))+"%")
    