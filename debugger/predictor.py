from  keras import models
import numpy as np

class Predictor:

    def __init__(self, model_path):
        self.model = models.load_model(model_path)
        self.model.summary()

    def predict(self, data):

        data = np.expand_dims(data, axis = 0)
        prediction = self.model.predict(data)
        return prediction[0]



