from sklearn.base import BaseEstimator, ClassifierMixin


class DummyFailureModel(BaseEstimator, ClassifierMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return [0] * len(X)
