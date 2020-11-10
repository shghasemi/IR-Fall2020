import pandas as pd
from processor import EnglishProcessor, PersianProcessor
import os
import xml.etree.ElementTree as ElementTree
from PositionalIndex import PositionalIndex
from tf_idf_search import tf_idf_search
from edit_query import edit_query
from proximity_search import proximity_search


class InformationRetrieval():
    def __init__(self, lang):
        self.lang = lang
        self.processor = None
        self.processed_docs = None
        self.title_processed_docs = None
        if self.lang == 'english':
            self.read_english_data()
            self.pi = PositionalIndex('description english pi', self.processed_docs)
            self.tpi = PositionalIndex('title english pi', self.title_processed_docs)
            self.tpi.show_posting_list('man')
        else:
            self.read_persian_data()
            self.pi = PositionalIndex('persian pi', self.processed_docs)
            self.tpi = PositionalIndex('title persian pi', self.title_processed_docs)

    def read_persian_data(self):
        docs, titles = self.read_xml('./data/Persian.xml', '{http://www.mediawiki.org/xml/export-0.10/}')
        self.processor = PersianProcessor()
        self.processed_docs = self.processor.process_docs(docs)
        self.title_processed_docs = self.processor.process_docs(titles, find_stopwords=False)

    def read_english_data(self):
        doc = pd.read_csv(os.path.join('data', 'ted_talks.csv'))
        self.processor = EnglishProcessor()
        self.processed_docs = self.processor.process_docs(doc['description'])
        self.title_processed_docs = self.processor.process_docs(doc['title'], find_stopwords=False)

    def read_xml(self, path, namespace):
        root = ElementTree.parse(path).getroot()
        docs = [page.find(f'{namespace}revision/{namespace}text').text for page in root.findall(namespace + 'page')]
        titles = [page.find(f'{namespace}revision/{namespace}title').text for page in root.findall(namespace + 'page')]
        return docs, titles


if __name__ == '__main__':
    ir = InformationRetrieval('english')
    # ir = InformationRetrieval('persian')
    # print(ir.processor.stopwords_freq)
    doc_id_list = list(range(len(ir.processed_docs)))
    n = len(doc_id_list)
    k = 10
    w = 3

    # Edit query
    # print("Please enter a wrong query :)")
    # query = input()
    # processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    # edited_query = edit_query(ir.pi.index.keys(), processed_query)
    # print("Initial query: ", query)
    # print("Processed query: ", processed_query)
    # print("Edited query: ", edited_query)

    # tf_idf_search
    # print("Please enter a right query :)")
    # query = input()
    # query = 'بازی'
    # query = 'talk'
    # processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    # print(tf_idf_search(n, doc_id_list, processed_query, ir.pi.index, 10))

    # proximity_search
    # query = 'شاهد عینی'
    query = 'talks'
    processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    print(processed_query)

    retrieved = tf_idf_search(n, doc_id_list, processed_query, ir.pi.index, k)
    # retrieved = proximity_search(n, doc_id_list, processed_query, ir.pi.index, w, k)
    for i, (doc_id, score) in enumerate(retrieved):
        print(f'{i + 1:2d}. ID: {doc_id:5d}, Score: {score:.4f}')

