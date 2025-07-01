# model_base.py
class ModelBase:
    def train(self, X_train, y_train):
        raise NotImplementedError

    def predict(self, X):
        raise NotImplementedError
