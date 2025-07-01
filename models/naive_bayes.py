# naive_bayes.py
from sklearn.naive_bayes import MultinomialNB
from .model_base import ModelBase

class NaiveBayesModel(ModelBase):
    def __init__(self):
        self.model = MultinomialNB()

    def train(self, X_train, y_train):
        self.model.fit(X_train, y_train)

    def predict(self, X):
        return self.model.predict(X)
