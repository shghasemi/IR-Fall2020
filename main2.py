import numpy as np
import pandas as pd
from sklearn import metrics
import os

from sklearn.model_selection import train_test_split

from PositionalIndex import PositionalIndex
from processor import EnglishProcessor
from proximity_search import proximity_search
from tf_idf_search import idf_vector, tf_idf_search
from InformationRetrieval import InformationRetrieval
from classifier import NaiveBayesClassifier, KNNClassifier, SVMClassifier, RandomForest


def read_data(processor, path, file='ted_talks'):
    is_ted = file == 'ted_talks'
    doc = pd.read_csv(os.path.join(path, f'{file}.csv'))
    processed_docs = processor.process_docs(doc['description'], find_stopwords=is_ted)
    processed_titles = processor.process_docs(doc['title'], find_stopwords=False)
    doc_ids = np.array(list(range(len(processed_docs))))
    y = np.array(doc['views']) if not is_ted else None
    # positional index
    pi = PositionalIndex(f'{file}_pi', processed_docs, doc_ids)
    pi.build(processed_titles, doc_ids)
    return doc_ids, y, pi


def doc_vector_ntn(doc, dictionary, pos_idx, idf):
    tf = np.array([len(pos_idx.get(token, {}).get(doc, [])) for token in dictionary], dtype=np.float)
    return tf * idf


def doc2vec(doc_id_list, dictionary, pos_idx, idf):
    X = [doc_vector_ntn(doc, dictionary, pos_idx, idf) for doc in doc_id_list]
    return np.array(X)


def build_X(path):
    processor = EnglishProcessor()
    # Find dictionary
    ids_ted, _, pi_ted = read_data(processor, path=path, file='ted_talks')
    ids_train, y_train, pi_train = read_data(processor, path=path, file='train')
    ids_test, y_test, pi_test = read_data(processor, path=path, file='test')
    dictionary = pi_train.index.keys()
    # build ted, train, and test matrices
    idf = idf_vector(len(ids_train), dictionary, pi_train.index)
    X_train = doc2vec(ids_train, dictionary, pi_train.index, idf)
    X_test = doc2vec(ids_test, dictionary, pi_test.index, idf)
    X_ted = doc2vec(ids_ted, dictionary, pi_ted.index, idf)
    print(X_ted[X_ted != 0])
    return X_train, X_test, y_train, y_test, ids_ted, X_ted


def evaluate(y_true, y_pred):
    print(metrics.classification_report(y_true, y_pred))
    tn, fp, fn, tp = metrics.confusion_matrix(y_true, y_pred).ravel()
    precision_t = tp / (tp + fp)
    precision_n = tn / (tn + fn)
    precision_m = (precision_t + precision_n) / 2
    recall_t = tp / (tp + fn)
    recall_n = tn / (tn + fp)
    recall_m = (recall_t + recall_n) / 2
    f1_t = 2 * precision_t * recall_t / (precision_t + recall_t)
    f1_n = 2 * precision_n * recall_n / (precision_n + recall_n)
    f1_m = 2 * precision_m * recall_m / (precision_m + recall_m)
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    print('\t\tprecision\trecall\t\tf1-score\n')
    print(f' 1\t\t{precision_t:0.4f}\t\t{recall_t:0.4f}\t\t{f1_t:0.4f}')
    print(f'-1\t\t{precision_n:0.4f}\t\t{recall_n:0.4f}\t\t{f1_n:0.4f}')
    print(f'\naccuracy\t\t\t\t\t{accuracy:0.4f}')
    print(f'macro avg\t{precision_m:0.4f}\t\t{recall_m:0.4f}\t\t{f1_m:0.4f}')


def test_search(ir, doc_ids, proximity=False, search_title=False):
    # init IR system, doc ID list, N, and k
    n = len(ir.doc_ids)
    k = 10
    p_idx = ir.tpi.index if search_title else ir.pi.index
    docs = ir.title_processed_docs if search_title else ir.processed_docs

    # input query and window size
    query = input("Please enter a correct query: \n")
    if proximity:
        w = int(input("Please enter proximity window size: \n"))

    processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    print("Initial query: ", query)
    print("Processed query: ", processed_query)

    # retrieve docs
    if proximity:
        retrieved = proximity_search(n, doc_ids, processed_query, p_idx, w, k)
        print(f'Top {len(retrieved)} doc using proximity search - window size = {w}:')
    else:
        retrieved = tf_idf_search(n, doc_ids, processed_query, p_idx, k)
        print(f'Top {len(retrieved)} doc using tf-idf search:')

    # pring result
    for i, (doc_id, score) in enumerate(retrieved):
        print(f'{i + 1:2d}. ID: {doc_id:5d}, Score: {score:.5f}')
        print(docs[ir.doc_ids.index(doc_id)])


if __name__ == '__main__':
    datapath = "data"
    X_train_val, X_test, y_train_val, y_test, ted_ids, X_ted = build_X(datapath)
    print("X_train_val shape: {}, X_test shape: {}".format(X_train_val.shape, X_test.shape))
    print("y_train_val shape: {}, y_test shape: {}".format(y_train_val.shape, y_test.shape))
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=0.1, random_state=42)
    print("X_train shape: {}, X_val shape: {}".format(X_train.shape, X_val.shape))
    print("y_train shape: {}, y_val shape: {}".format(y_train.shape, y_val.shape))

    nb_clf = NaiveBayesClassifier()
    nb_clf.fit(X_train, y_train)
    y_pred_val = nb_clf.predict(X_val)
    y_pred_test = nb_clf.predict(X_test)
    print('NB validation acc: {}'.format((y_pred_val == y_val).mean()))
    evaluate(y_test, y_pred_test)

    for k in [1, 5, 9]:
        knn_clf = KNNClassifier(k)
        knn_clf.fit(X_train, y_train)
        y_pred_val = knn_clf.predict(X_val)
        y_pred_test = knn_clf.predict(X_test)
        print('{}-nn validation acc: {}'.format(k, (y_pred_val == y_val).mean()))
        evaluate(y_test, y_pred_test)

    c_values = [0.5, 1, 1.5, 2]
    for C in c_values:
        svm_clf = SVMClassifier(C=C)
        svm_clf.fit(X_train, y_train)
        y_pred_val = svm_clf.predict(X_val)
        y_pred_test = svm_clf.predict(X_test)
        print('svm-{} validation acc: {}'.format(C, (y_pred_val == y_val).mean()))
        evaluate(y_test, y_pred_test)

    random_forest_clf = RandomForest()
    random_forest_clf.fit(X_train, y_train)
    y_pred_val = random_forest_clf.predict(X_val)
    y_pred_test = random_forest_clf.predict(X_test)
    print('forest validation acc: {}'.format((y_pred_val == y_val).mean()))
    evaluate(y_test, y_pred_test)
