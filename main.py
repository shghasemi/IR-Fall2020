import pandas as pd
from processor import EnglishProcessor
import os


print("Please select language: 1:English 2:Persian")
lang = int(input())

if lang == 1:
    doc = pd.read_csv(os.path.join('data', 'ted_talks.csv'))
    processor = EnglishProcessor()
    final_docs, stop_words = processor.preprocess_docs(doc['description'])
    print("initial doc 0 : ", doc['description'][0])
    print("final doc 0 : ", final_docs[0])
    print("Stop words : ", stop_words)
