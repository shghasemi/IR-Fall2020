
from tf_idf_search import tf_idf_search
from proximity_search import proximity_search


if __name__ == '__main__':
    ir = InformationRetrieval('persian')
    print(ir.processor.stopwords_freq)
    doc_id_list = ir.doc_ids
    n = len(doc_id_list)
    k = 10
    w = 3
    query = 'امتداد کوه'
    processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    print(processed_query)
    retrieved = tf_idf_search(n, doc_id_list, processed_query, ir.pi.index, k)
    print(f'Top {len(retrieved)} doc using tf-idf search:')
    for i, (doc_id, score) in enumerate(retrieved):
        print(f'{i + 1:2d}. ID: {doc_id:5d}, Score: {score:.4f}')
        print(ir.processed_docs[ir.doc_ids.index(doc_id)])
    retrieved = proximity_search(n, doc_id_list, processed_query, ir.pi.index, w, k)
    print(f'Top {len(retrieved)} doc using proximity search - window size = {w}:')
    for i, (doc_id, score) in enumerate(retrieved):
        print(f'{i + 1:2d}. ID: {doc_id:5d}, Score: {score:.4f}')
        print(ir.processed_docs[ir.doc_ids.index(doc_id)])
