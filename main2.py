import numpy as np
import pandas as pd
from sklearn import metrics
import os

from PositionalIndex import PositionalIndex
from processor import EnglishProcessor
from tf_idf_search import idf_vector


def read_data(processor, file='ted_talks'):
    is_ted = file == 'ted_talks'
    doc = pd.read_csv(os.path.join('data', f'{file}.csv'))
    processed_docs = processor.process_docs(doc['description'], find_stopwords=is_ted)
    processed_titles = processor.process_docs(doc['title'], find_stopwords=False)
    doc_ids = list(range(len(processed_docs)))
    y = doc['views'] if not is_ted else None
    # positional index
    pi = PositionalIndex(f'{file}_pi', processed_docs, doc_ids)
    pi.build(processed_titles, doc_ids)
    return doc_ids, y, pi


def doc_vector_ntn(doc, dictionary, pos_idx, idf):
    tf = np.array([len(pos_idx.get(token, {}).get(doc, [])) for token in dictionary], dtype=np.float)
    return tf * idf


def doc2vec(doc_id_list, dictionary, pos_idx):
    n = len(doc_id_list)
    idf = idf_vector(n, dictionary, pos_idx)
    X = [doc_vector_ntn(doc, dictionary, pos_idx, idf) for doc in doc_id_list]
    return np.array(X)


def build_X():
    processor = EnglishProcessor()
    # Find dictionary
    _, _, pi_complete = read_data(processor, 'ted_talks')
    dictionary = pi_complete.index.keys()
    # build train and test matrices
    ids_train, y_train, pi_train = read_data(processor, 'train')
    ids_test, y_test, pi_test = read_data(processor, 'test')
    X_train = doc2vec(ids_train, dictionary, pi_train.index)
    X_test = doc2vec(ids_test, dictionary, pi_test.index)
    return X_train, X_test


def evaluate(y_true, y_pred):
    accuracy = np.mean(y_pred == y_true)
    print(metrics.confusion_matrix(y_true, y_pred))
    print(metrics.classification_report(y_true, y_pred, digits=3))
    # TP = np.sum(np.logical_and(y_true == 1, y_pred == 1))
    # FP = np.sum(np.logical_and(y_true == 0, y_pred == 1))
    # TN = np.sum(np.logical_and(y_true == 0, y_pred == 0))
    # FN = np.sum(np.logical_and(y_true == 1, y_pred == 0))
    # specifity = TN / (TN + FP)
    # sensitivity = TP / (TP + FN)


# if __name__ == '__main__':
#     X_train, y_train = read_text_file('data/phase2_train.csv')
#     X_test, y_test = read_text_file('data/phase2_test.csv')
#
#     print("Please select classifier")
#     print("1. Naive Bayes")
#     print("2. k-NN")
#     print("3. SVM")
#     print("4. Random Forest")
#     cls_number = int(input())
#     set_hypers = True if cls_number >= 3 else False
#     if cls_number == 2:
#         learn_k(20, X_train, y_train)
#     cls = get_classifier(cls_number=cls_number)
#
#     # build a pipeline using cls
#     text_cls, search = build_pipeline(cls, set_hyperparams=set_hypers)
#     if set_hypers:
#         search.fit(X_train, y_train)
#     # print("fit done")
#     text_cls.fit(X_train, y_train)
#     y_train_perd = text_cls.predict(X_train)
#     y_test_pred = text_cls.predict(X_test)
#     print("Evaluating train data ================")
#     evaluate(y_train, y_train_perd)
#     print("Evaluating test data =================")
#     evaluate(y_test, y_test_pred)
#
#     # tfidf_transformer = doc2vec(docs)
#     # X_train = tfidf_transformer.transform(docs)
#     # X_test = tfidf_transformer.transform(docs_test)

if __name__ == '__main__':
    X_train, X_test = build_X()
    print(X_train[1:5, ])
    print(X_train.shape, X_test.shape)
