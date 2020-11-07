class BiGramIndex:
    def __init__(self, name, docs):
        self.name = name
        self.index = {}
        self.build(docs)

    def build(self, docs):
        """
        :param docs: list of docs
        each doc is a list of tokens
        :return:
        """
        for doc_id, doc in enumerate(docs):
            for token in doc:
                self.add_token(doc_id, token)

    def add_token(self, doc_id, token):
        bounded_token = '$' + token + '$'
        for i in range(len(bounded_token) - 1):
            term = bounded_token[i: i + 2]
            if term not in self.index:
                self.index[term] = dict()
            if token not in self.index[term]:
                self.index[term][token] = []
            if doc_id not in self.index[term][token]:
                self.index[term][token].append(doc_id)

    def delete_doc(self, doc_id):
        for term in self.index.keys():
            for token in self.index[term].keys():
                self.index[term][token].remove(doc_id)

    def show_tokens_contain_bigram(self, bigram):
        print(self.index[bigram].keys())


# bi = BiGramIndex("test",
#                      [["hello", "world", "!"], ["fuck", "you"], ["hello"], ["ololo"]])
