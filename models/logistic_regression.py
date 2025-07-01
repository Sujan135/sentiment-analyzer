from sklearn.linear_model import LogisticRegression
from .model_base import ModelBase

class LogisticRegressionModel(ModelBase):
    def __init__(self):
        self.model = LogisticRegression(max_iter=1000, solver='liblinear')

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
