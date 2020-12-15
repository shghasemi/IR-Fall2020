from PositionalIndex import PositionalIndex
import pandas as pd
from processor import EnglishProcessor, PersianProcessor
import os
import xml.etree.ElementTree as ElementTree


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

