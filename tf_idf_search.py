import numpy as np


def tf_vector(query, dictionary):
    vec = np.zeros(len(dictionary))
    for i, token in enumerate(query):
        if token in dictionary:
            vec[dictionary.index(token)] += 1
    return vec


def log_tf_vector(query, dictionary):
    vec = tf_vector(query, dictionary)
    non_zero = vec != 0
    vec[non_zero] = 1 + np.log(vec[non_zero])
    return vec


def log_tf_vector_doc(doc_id, dictionary, pos_idx):
    vec = np.array([len(pos_idx[token].get(doc_id, [])) for token in dictionary], dtype=np.float)
    non_zero = vec != 0
    vec[non_zero] = 1 + np.log(vec[non_zero])
    return vec


def idf_vector(n, dictionary, pos_idx):
    vec = np.array([len(pos_idx[token]) for token in dictionary], dtype=np.float)
    non_zero = vec != 0
    if n:
        vec[non_zero] = np.log(n / vec[non_zero])
    return vec


def normalize_vector(vec):
    c = np.sum(vec * vec)
    return vec / c if c else vec


def query_vector(query, dictionary):
    return normalize_vector(log_tf_vector(query, dictionary))


def doc_vector(doc, dictionary, pos_idx, idf):
    tf = log_tf_vector_doc(doc, dictionary, pos_idx)
    return normalize_vector(tf * idf)


def tf_idf_search(n, doc_id_list, query, pos_idx, k):
    dictionary = list(pos_idx.keys())
    q_vec = query_vector(query, dictionary)
    idf = idf_vector(n, dictionary, pos_idx)
    scores = []
    for doc_id in doc_id_list:
        d_vec = doc_vector(doc_id, dictionary, pos_idx, idf)
        scores.append((doc_id, q_vec.dot(d_vec)))
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_docs[:min(k, len(ranked_docs))]
