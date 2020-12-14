import pandas as pd

from processor import EnglishProcessor, PersianProcessor
import os
import xml.etree.ElementTree as ElementTree
from PositionalIndex import PositionalIndex
from tf_idf_search import tf_idf_search
from proximity_search import proximity_search


class InformationRetrieval:
    def __init__(self, lang):
        self.lang = lang
        self.processor = None
        self.processed_docs = None
        self.title_processed_docs = None
        self.doc_ids = None
        if self.lang == 'english':
            self.read_english_data()
        else:
            self.read_persian_data()
        # positional index
        self.pi = PositionalIndex('description ' + self.lang + ' pi', self.processed_docs, self.doc_ids)
        self.tpi = PositionalIndex('title ' + self.lang + ' pi', self.title_processed_docs, self.doc_ids)

    def read_persian_data(self):
        docs, titles, ids = self.read_xml('./data/Persian.xml', '{http://www.mediawiki.org/xml/export-0.10/}')
        self.processor = PersianProcessor()
        self.processed_docs = self.processor.process_docs(docs)
        self.title_processed_docs = self.processor.process_docs(titles, find_stopwords=False)
        self.doc_ids = ids

    def read_english_data(self):
        doc = pd.read_csv(os.path.join('data', 'ted_talks.csv'))
        self.processor = EnglishProcessor()
        self.processed_docs = self.processor.process_docs(doc['description'])
        self.title_processed_docs = self.processor.process_docs(doc['title'], find_stopwords=False)
        self.doc_ids = list(range(len(self.processed_docs)))

    def read_xml(self, path, namespace):
        root = ElementTree.parse(path).getroot()
        docs = [page.find(f'{namespace}revision/{namespace}text').text for page in root.findall(namespace + 'page')]
        titles = [page.find(f'{namespace}title').text for page in root.findall(namespace + 'page')]
        # ids = list(range(len(docs)))
        ids = [int(page.find(f'{namespace}id').text) for page in root.findall(namespace + 'page')]
        return docs, titles, ids


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
