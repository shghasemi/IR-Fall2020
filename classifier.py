import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from math import log2


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

    def __init__(self):
        self.name = 'Naive Bayes'
        self.prob_c0 = 0
        self.prob_c1 = 0
        self.prob_terms_c0 = None
        self.prob_terms_c1 = None

    def fit(self, X_train, y_train):
        total_num_docs = X_train.shape[0]
        num_distinct_terms = X_train.shape[1]
        self.prob_c0 = sum(y_train[y_train > 0]) / total_num_docs
        self.prob_c1 = 1 - self.prob_c0

        num_docs_per_term_c0 = []
        num_docs_per_term_c1 = []
        total_num_terms_c0 = 0
        total_num_terms_c1 = 0
        for column in X_train.T:
            num_term_c0 = sum(column[y_train > 0])
            num_term_c1 = sum(column) - num_term_c0
            num_docs_per_term_c0.append(num_term_c0 + 1)
            num_docs_per_term_c1.append(num_term_c1 + 1)
            total_num_terms_c0 += num_term_c0
            total_num_terms_c0 += num_term_c1
        self.prob_terms_c0 = np.array([log2(f / (num_distinct_terms + total_num_terms_c0))
                                       for f in num_docs_per_term_c0])
        self.prob_terms_c1 = np.array([log2(f / (num_distinct_terms + total_num_terms_c1))
                                       for f in num_docs_per_term_c1])

    def predict_doc_label(self, doc):
        return 1 if log2(self.prob_c0) + sum(self.prob_terms_c0[doc > 0]) >= log2(self.prob_c1) + sum(self.prob_terms_c1[doc > 0]) else -1

    def predict(self, test):
        labels = []
        for doc in test:
            labels.append(self.predict_doc_label(doc))
        return np.array(labels)

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




