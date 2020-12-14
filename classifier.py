import numpy as np


class KNNClassifier:
    def __init__(self, k, X_train, y_train):
        self.k = k
        self.X_train = X_train
        self.y_train = y_train

    @staticmethod
    def majority(lst):
        return max(set(lst), key=lst.count)

    @staticmethod
    def distance(v1, v2):
        return abs(np.linalg.norm(v2 - v1, 2))

    def neighbors(self, x):
        d = [(self.y_train[i], self.distance(self.X_train[i], x)) for i in range(self.X_train.shape[0])]
        d.sort(key=lambda t: t[1])
        return [d[i][0] for i in range(self.k)]

    def predict(self, X_test):
        y_pred = [self.majority(self.neighbors(x)) for x in X_test]
        return np.array(y_pred)
