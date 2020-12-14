import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


class KNNClassifier:
    def __init__(self, k):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

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


class NaiveBayesClassifier:
    pass

class SVMClassifier:
    def __init__(self, C):
        self.C = C
        self.X_train = None
        self.y_train = None
        self.clf = SVC(C=C)

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.clf.fit(X, y)

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        return y_pred

class RandomForest:
    def __init__(self):
        self.X_train = None
        self.y_train = None
        self.clf = RandomForestClassifier()
        
    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.clf.fit(X, y)

    def predict(self, X_test):
        y_pred = self.clf.predict(X_test)
        return y_pred




