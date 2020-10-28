import json
import os
from matplotlib import pyplot as plt

class Trainer:

    def __init__(self, model, x_train, y_train, x_val, y_val):
        self.model = model
        self.x_train = x_train
        self.y_train = y_train
        self.x_val = x_val
        self.y_val = y_val

    def train(self, epochs, batch_size, save_dir):

        history = self.model.fit(
                self.x_train,
                self.y_train,
                epochs = epochs,
                batch_size = batch_size,
                validation_data = (self.x_val, self.y_val)
                )

        self.save(history, save_dir)
        self._plot(history.history, save_dir)

    def save(self, history, save_dir):

        if os.path.isdir(save_dir) == False:
            os.mkdir(save_dir)

        train_records = {}
        for key , value in history.history.items():
            train_records[key] = []

            for data in value:
                train_records[key].append(float(data))

        json_path = os.path.join(save_dir, "train_records.json")

        with open(json_path, "w") as f:
            json.dump(train_records, f)

        model_path = os.path.join(save_dir, "model.h5")
        self.model.save(model_path)


    def _plot(self, data, save_dir):

        self._plot_loss(data, save_dir)
        self._plot_acc(data, save_dir)


    def _plot_loss(self, data, save_dir):

        loss_values = data["loss"]
        val_loss_values = data["val_loss"]
        epochs = range(1, len(loss_values) + 1)

        plt.plot(epochs, loss_values, "bo", label = "Training loss")
        plt.plot(epochs, val_loss_values, "b", label = "Validation loss")

        plt.title("Training and validation loss")
        plt.xlabel("Epochs")
        plt.ylabel("Loss")
        plt.legend()

        plt.savefig(os.path.join(save_dir, "loss.png"))
        plt.clf()


    def _plot_acc(self, data, save_dir):

        acc_values = data["accuracy"]
        val_acc_values = data["val_accuracy"]
        epochs = range(1, len(acc_values) + 1)

        plt.plot(epochs, acc_values, "bo", label = "Training accuracy")
        plt.plot(epochs, val_acc_values, "b", label = "Validation accuracy")

        plt.title("Training and validation accuracy")
        plt.xlabel("Epochs")
        plt.ylabel("accuracy")
        plt.legend()

        plt.savefig(os.path.join(save_dir, "accuracy.png"))
        plt.clf()




















