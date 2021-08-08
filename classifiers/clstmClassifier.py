from keras.models import Sequential
from keras import layers

class ClstmClassifier:
    def __init__(self):
        pass
    # init


    def train_model(self, vocab_size, max_len, X_train, y_train, X_test, y_test):
        lstm_units = 40
        embedding_dim = 50
        conv_kernel_size = 7
        conv_filters_num = 115

        print("Training model...")

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

        model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=15)

        model.save("data/models/clstm.h5")
        print("Done")
        return model

    # end

