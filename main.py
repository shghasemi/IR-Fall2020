import pandas as pd
from processor import EnglishProcessor, PersianProcessor
import os
import xml.etree.ElementTree as ElementTree
from PositionalIndex import PositionalIndex
from tf_idf_search import tf_idf_search
from edit_query import edit_query


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
    # print(ir.processor.stopwords_freq)



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
    query = 'بازی'
    processed_query = ir.processor.process_docs([query], find_stopwords=False)[0]
    print(tf_idf_search(10, processed_query, ir.processed_docs, ir.pi.index))

