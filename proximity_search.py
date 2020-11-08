import itertools
from tf_idf_search import tf_idf_search


def contains_query(doc_id, query, pos_idx):
    return all(doc_id in pos_idx[token] for token in query)


def is_relevant(doc_id, query, pos_idx, window):
    if not contains_query(doc_id, query, pos_idx):
        return False
    pos_list = [pos_idx[token][doc_id] for token in query]
    pos_comb = list(itertools.product(*pos_list))
    return any(max(pos) - min(pos) < window for pos in pos_comb)


def proximity_search(n, doc_id_list, query, pos_idx, window, k):
    docs = [doc_id for doc_id in doc_id_list if is_relevant(doc_id, query, pos_idx, window)]
    return tf_idf_search(n, docs, query, pos_idx, k)
