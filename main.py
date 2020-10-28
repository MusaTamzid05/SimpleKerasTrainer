from keras.datasets import imdb
from keras import layers
from keras import models
import numpy as np
import json
from debugger.trainer import Trainer

def vectorizer_sequence(sequences, dimension = 10000):

    results = np.zeros((len(sequences), dimension))

    for i, sequence  in enumerate(sequences):
        results[i, sequence] = 1

    return results

def init_model():

    model = models.Sequential()
    model.add(layers.Dense(64, activation = "relu",  input_shape = (10000,)))
    model.add(layers.Dense(32, activation = "relu"))
    model.add(layers.Dense(16, activation = "relu"))
    model.add(layers.Dense(8, activation = "relu"))
    model.add(layers.Dense(4, activation = "relu"))
    model.add(layers.Dense(2, activation = "relu"))
    model.add(layers.Dense(1, activation = "sigmoid"))

    model.compile(optimizer = "rmsprop",
            loss = "binary_crossentropy",
            metrics = ["accuracy"]
            )

    return model

def main():
    (train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = 10000)

    model = init_model()
    model.summary()

    x_train = vectorizer_sequence(train_data)
    x_test = vectorizer_sequence(test_data)

    y_train = np.array(train_labels).astype("float32")
    y_test = np.array(test_labels).astype("float32")

    x_val = x_train[:10000]
    partial_x_train = x_train[10000:]

    y_val = y_train[:10000]
    partial_y_train = y_train[10000:]

    trainer = Trainer(model = model,
            x_train = partial_x_train,
            y_train = partial_y_train,
            x_val = x_val,
            y_val = y_val
            )

    trainer.train(epochs = 20, batch_size = 128, save_dir = "model_data")




if __name__ == "__main__":
    main()
