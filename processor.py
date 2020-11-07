import nltk
import re
from collections import defaultdict
from nltk.stem import PorterStemmer


class EnglishProcessor:

    def normalize(self, sentence):

        case_folded = sentence.lower()
        no_number = ''
        for i in range(len(case_folded)):
            if sentence[i] in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']:
                continue
            else:
                no_number += case_folded[i]
        no_whitespace = no_number.strip()
        no_punc = self.remove_punctuations(no_whitespace)
        token_list = nltk.word_tokenize(no_punc)
        stemmed_list = self.stem(token_list)
        return stemmed_list


    def remove_punctuations(self, sentence):
        return re.sub(r'[^\w\s]', '', sentence)

    def stem(self, token_list):
        stemmer = PorterStemmer(PorterStemmer.ORIGINAL_ALGORITHM)
        return [stemmer.stem(x) for x in token_list]

    def preprocess_docs(self, docs):
        normalized_docs = [self.normalize(doc) for doc in docs]

        stop_words = self.find_stop_words(normalized_docs)

        final_docs = [self.remove_stopwords(stop_words, normalized_doc) for normalized_doc in normalized_docs]

        return final_docs, stop_words

    def find_stop_words(self, normalized_docs):

        total_word_count = 0
        word_freq = {}
        word_freq = defaultdict(lambda: 0, word_freq)
        for i in range(len(normalized_docs)):
            doc = normalized_docs[i]
            for j in range(len(doc)):
                word = doc[j]
                word_freq[word] += 1
                total_word_count += 1

        stop_word_count = total_word_count // 100
        stop_words = [word for word, count in word_freq.items() if count > stop_word_count]

        print("Total word count = {}, stop word count threshold = {}".format(total_word_count, stop_word_count))
        return stop_words

    def remove_stopwords(self, stopwords, tokens):
        return [x for x in tokens if x not in stopwords]
