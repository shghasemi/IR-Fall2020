import pandas as pd
from processor import EnglishProcessor
from processor import PersianProcessor
import os
import xml.etree.ElementTree as ElementTree


def read_xml(path, namespace):
    root = ElementTree.parse(path).getroot()
    docs = [page.find(namespace + 'revision/' + namespace + 'text').text for page in root.findall(namespace + 'page')]
    return docs

print("Please select language: 1:English 2:Persian")
lang = int(input())

if lang == 1:
    doc = pd.read_csv(os.path.join('data', 'ted_talks.csv'))
    processor = EnglishProcessor()
    processed_docs, stopwords = processor.process_docs(doc['description'])
    print("initial doc 0 : ", doc['description'][0])
    print("final doc 0 : ", processed_docs[0])
    print("Stop words : ", stopwords)

if lang == 2:
    docs = read_xml('./data/Persian.xml', '{http://www.mediawiki.org/xml/export-0.10/}')
    processor = PersianProcessor()
    processed_docs, stopwords = processor.process_docs(docs)  # [:10])
    print("processed doc 0 : ", processed_docs[0])
    print('Persian stopwords: {}'.format(stopwords))


# if __name__ == '__main__':
