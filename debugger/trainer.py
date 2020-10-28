import json
import os


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




