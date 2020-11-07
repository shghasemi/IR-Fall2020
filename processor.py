import nltk
import hazm
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
        return [stemmer.stem(token) for token in token_list]

    def process_docs(self, docs):
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
        stop_words = [(word, count) for word, count in word_freq.items() if count > stop_word_count]

        print("Total word count = {}, stop word count threshold = {}".format(total_word_count, stop_word_count))
        return stop_words

    def remove_stopwords(self, stopwords, token_list):
        return [token for token in token_list if token not in stopwords]


class PersianProcessor:

    def normalize(self, sentence):
        return self.stem(self.tokenize(self.remove_puncts(hazm.Normalizer().normalize(sentence))))

    def tokenize(self, sentence):
        return hazm.word_tokenize(sentence)

    def remove_puncts(self, sentence):
        return re.sub(r'[^\w\s]', '', re.sub(r'[a-zA-Z_]', '', re.sub(r'[۰-۹0-9]', ' ', sentence)))

    def stem(self, token_list):
        stemmer = hazm.Stemmer()
        return [stemmer.stem(token) for token in token_list]

    def remove_stopwords(self, token_list, stopwords):
        return [token for token in token_list if token not in stopwords]

    def process_docs(self, docs):
        processed_docs = [self.normalize(doc) for doc in docs]
        stopwords = self.find_stopwords(processed_docs)
        processed_docs = [self.remove_stopwords(doc, stopwords.keys()) for doc in processed_docs]
        return processed_docs, stopwords

    def find_stopwords(self, docs):
        word_freq = {}
        for doc in docs:
            for word in doc:
                word_freq[word] = 1 if word not in word_freq else word_freq[word] + 1
        thr = sum(word_freq.values()) * 0.008
        stopwords = {word: freq for word, freq in word_freq.items() if freq >= thr}
        return stopwords
