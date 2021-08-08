import numpy as np
import tensorflow as tf
from keras import layers
from keras import Sequential
#from tensorflow import keras
#from tensorflow.keras import layers
import pandas as pd
from feature_extract import FeatureExtractor
f = FeatureExtractor()
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt
plt.style.use('ggplot')
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from keras.preprocessing.sequence import pad_sequences


def plot_history(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    x = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(x, acc, 'b', label='Training acc')
    plt.plot(x, val_acc, 'r', label='Validation acc')
    plt.title('Training and validation accuracy')
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.plot(x, loss, 'b', label='Training loss')
    plt.plot(x, val_loss, 'r', label='Validation loss')
    plt.title('Training and validation loss')
    plt.legend()


    # preprocess data
def preprocess_data():
    labels = ["bearish", "bullish", "neutral"]
    trainDir = "./data/train/"
    dict_list = []
    for label in labels:
        file = open(trainDir + label + ".txt", 'r')
        for line in file:
            tweet = f.clean_tweet(line)
            dict_list.append({"text" : tweet, "label" : label})
            #print({line : label})
    df = pd.DataFrame(dict_list)

    tweets = df['text'].values
    y = df['label'].values
    tweets_train, tweets_test, y_train, y_test = train_test_split(tweets, y, test_size=0.20, random_state=1000)
    print(tweets_train.shape[0], "training tweets,", tweets_test.shape[0], "testing tweets")

    print(type(tweets_train))

    vectorizer = CountVectorizer(strip_accents='ascii', stop_words='english', ngram_range=(2,2))
    vectorizer.fit(tweets_train)

    X_train = vectorizer.transform(tweets_train)
    X_test  = vectorizer.transform(tweets_test)


    enc = preprocessing.OneHotEncoder()
    y_train = y_train.reshape(-1,1)
    y_train = enc.fit_transform(y_train).toarray()
    y_test = y_test.reshape(-1,1)
    y_test = enc.fit_transform(y_test).toarray()

    # X now numpy array of arrays for each tweet with counts of each bigram
    # y now numpy array of one-hot arrays representing 

    # create word embeddings
    from keras.preprocessing.text import Tokenizer

    tokenizer = Tokenizer(num_words=5000)
    tokenizer.fit_on_texts(tweets_train)

    X_train = tokenizer.texts_to_sequences(tweets_train)
    X_test = tokenizer.texts_to_sequences(tweets_test)

    vocab_size = len(tokenizer.word_index) + 1      # add 1 because 0 is reserved and has no associated word

    # find longest array length to pad all arrays to below
    max_len = 0
    for arr in X_train:
        if len(arr) > max_len:
            max_len = len(arr)
    print(max_len)

    # pad text_to_sequences arrays with 0s

    X_train = pad_sequences(X_train, padding='post', maxlen=max_len)
    X_test = pad_sequences(X_test, padding='post', maxlen=max_len)

    print(X_train[2])

    return X_train, y_train, X_test, y_test, max_len, vocab_size




# hyperparam optimization
def make_model(lstm_units, conv_filters_num, conv_kernel_size, vocab_size, embedding_dim, max_len):
    model = Sequential()
    model.add(layers.Embedding(vocab_size, embedding_dim, input_length=max_len))
    # conv1D to identify features in the tweets
    model.add(layers.Conv1D(conv_filters_num, conv_kernel_size, activation='relu'))
    # bidir lstm to look at features over the whole tweet both ways
    model.add(layers.Bidirectional(layers.LSTM(lstm_units, activation='relu', return_sequences=True)))
    model.add(layers.Flatten())
    model.add(layers.Dense(10, activation='relu'))
    model.add(layers.Dense(3, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics='accuracy')
    #model.summary()
    return model




if __name__ == "__main__":

    X_train, y_train, X_test, y_test, max_len, vocab_size = preprocess_data()    

    epochs = 8
    batch_size = 10
    
    # {'conv_filters_num': 120, 'conv_kernel_size': 5, 'embedding_dim': 50, 'lstm_units': 42, 'max_len': 30, 'vocab_size': 2457}
    # 0.731483668088913

    param_grid = dict(lstm_units = [42,43,44],
                    conv_filters_num = [120],
                    conv_kernel_size = [5,6,7],
                    vocab_size = [vocab_size],
                    embedding_dim = [50],
                    max_len = [max_len])

    model = KerasClassifier(build_fn=make_model, epochs=epochs, batch_size=batch_size, verbose=True)
    grid = GridSearchCV(estimator=model, param_grid=param_grid, cv=4, verbose=True)#, n_iter=15)

    print("\n\nRunning grid search")

    grid_result = grid.fit(X_train, y_train)

    print(grid_result.best_params_)
    print(grid_result.best_score_)
