import numpy as np
from math import log2

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
        self.prob_c1 = 1 - self.prob_terms_c0

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
        return 1 if self.prob_c0 + sum(self.prob_terms_c0[doc > 0]) >= self.prob_c0 + sum(self.prob_terms_c0[doc > 0]) else -1

    