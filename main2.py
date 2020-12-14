import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn import metrics
from classifier import get_classifier, find_hyper_params, learn_k


def read_text_file(file_name):
    path = file_name

    with open(path) as f:
        df = pd.read_csv(path)
    # texts = [simple_preprocess(t) for t in df['Text'].values.tolist()]
    docs = df['Text'].values.tolist()  # TODO: preprocess text
    labels = df['Tag'].values.tolist()
    return docs, labels


def doc2vec(docs):
    cv = CountVectorizer()
    count_vector = cv.fit_transform(docs)
    tfidf_transformer = TfidfTransformer(smooth_idf=True, use_idf=True)
    # X_train = tfidf_transformer.fit_transform(count_vector)
    tfidf_transformer.fit(count_vector)
    return tfidf_transformer

    # df_idf = pd.DataFrame(tfidf_transformer.idf_, index=cv.get_feature_names(), columns=['idf_weights'])
    # df_idf.sort_values(by=['idf_weights']) # sort ascending

    ## to check ##
    # count_vector = cv.transform(docs)  # count mat
    # tf_idf_vector = tfidf_transformer.transform(count_vector)

    # feature_names = cv.get_feature_names()  # feature names
    # first_document_vector = tf_idf_vector[0]
    # df = pd.DataFrame(first_document_vector.T.todense(), index=feature_names, columns=['tfidf'])
    # df.sort_values(by=['tfidf'], ascending=False)


# def doc2vec(docs):
#     tfidf_vectorizer = TfidfVectorizer(use_idf=True)
#     tfidf_vectorizer_vectors = tfidf_vectorizer.fit_transform(docs)


# def build_doc2vec(model_name, vector_size, min_count, window, epochs):
#     model = Doc2Vec(vector_size=vector_size, min_count=min_count, epochs=epochs, window=window)
#

def build_pipeline(clf, set_hyperparams=False):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', clf)
                         ])
    search = None
    if set_hyperparams:
        search = find_hyper_params(clf_pipe=text_clf)
    return text_clf, search


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


if __name__ == '__main__':
    X_train, y_train = read_text_file('data/phase2_train.csv')
    X_test, y_test = read_text_file('data/phase2_test.csv')

    print("Please select classifier")
    print("1. Naive Bayes")
    print("2. k-NN")
    print("3. SVM")
    print("4. Random Forest")
    cls_number = int(input())
    set_hypers = True if cls_number >= 3 else False
    if cls_number == 2:
        learn_k(20, X_train, y_train)
    cls = get_classifier(cls_number=cls_number)

    # build a pipeline using cls
    text_cls, search = build_pipeline(cls, set_hyperparams=set_hypers)
    if set_hypers:
        search.fit(X_train, y_train)
    # print("fit done")
    text_cls.fit(X_train, y_train)
    y_train_perd = text_cls.predict(X_train)
    y_test_pred = text_cls.predict(X_test)
    print("Evaluating train data ================")
    evaluate(y_train, y_train_perd)
    print("Evaluating test data =================")
    evaluate(y_test, y_test_pred)

    # tfidf_transformer = doc2vec(docs)
    # X_train = tfidf_transformer.transform(docs)
    # X_test = tfidf_transformer.transform(docs_test)
