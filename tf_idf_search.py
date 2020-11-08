import numpy as np


def tf_vector(tokens, dictionary):
    vec = np.zeros(len(dictionary))
    for i, token in enumerate(tokens):
        if token in dictionary:
            vec[dictionary.index(token)] += 1
    return vec


def log_tf_vector(tokens, dictionary):
    vec = tf_vector(tokens, dictionary)
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
    vec[non_zero] = np.log(n / vec[non_zero])
    return vec


def normalize_vector(vec):
    c = np.sum(vec * vec)
    return vec / c if c else vec


def query_vector(opt_query, n, dictionary, pos_idx):
    tf = log_tf_vector(opt_query, dictionary)
    idf = idf_vector(n, dictionary, pos_idx)
    return normalize_vector(tf * idf)


def doc_vector(doc, dictionary, pos_idx):
    return normalize_vector(log_tf_vector_doc(doc, dictionary, pos_idx))


def tf_idf_search(k, opt_query, processed_docs, pos_idx):
    n = len(processed_docs)
    dictionary = list(pos_idx.keys())
    q_vec = query_vector(opt_query, n, dictionary, pos_idx)
    scores = []
    for doc_id in range(len(processed_docs)):
        d_vec = doc_vector(doc_id, dictionary, pos_idx)
        scores.append((doc_id, q_vec.dot(d_vec)))
    ranked_docs = sorted(scores, key=lambda x: x[1], reverse=True)
    return ranked_docs[:min(k, len(ranked_docs))]
