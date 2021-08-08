from keras.models import Sequential
from keras import layers

class ConvClassifier:
    def __init__(self):
        pass
    # init
    

    def train_model(self, vocab_size, max_len, X_train, y_train, X_test, y_test):
        
        model = Sequential()
        model.add(layers.Embedding(vocab_size, 50, input_length=max_len))
        model.add(layers.Conv1D(128, 4, activation='relu'))
        model.add(layers.GlobalMaxPool1D())
        model.add(layers.Dense(10, activation='relu')) # 4340*10= 43400 weights + 10 bias = 43410 params
        model.add(layers.Dense(3, activation='softmax'))

        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        
        model.fit(X_train, y_train, epochs=100, verbose=False, validation_data=(X_test, y_test), batch_size=15)

        model.save("data/models/cnn.h5")
        print("Done")
        return model

    # train_model



