import pandas as pd
from processor import EnglishProcessor
from processor import PersianProcessor
import os
import xml.etree.ElementTree as ElementTree
from PositionalIndex import PositionalIndex


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

    def read_persian_data(self):
        docs = self.read_xml('./data/Persian.xml', '{http://www.mediawiki.org/xml/export-0.10/}')
        self.processor = PersianProcessor()
        self.processed_docs = self.processor.process_docs(docs)

    def read_english_data(self):
        doc = pd.read_csv(os.path.join('data', 'ted_talks.csv'))
        self.processor = EnglishProcessor()
        self.processed_docs = self.processor.process_docs(doc['description'])
        self.title_processed_docs = self.processor.process_docs(doc['title'], find_stopwords=False)

    def read_xml(self, path, namespace):
        root = ElementTree.parse(path).getroot()
        docs = [page.find(f'{namespace}revision/{namespace}text').text for page in root.findall(namespace + 'page')]
        return docs


if __name__ == '__main__':
    ir = InformationRetrieval('persian')
    print(ir.processor.stopwords_freq)
