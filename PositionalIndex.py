class PositionalIndex:
    def __init__(self, name, docs):
        self.index = dict()
        self.name = name
        self.build(docs)

    def build(self, docs):
        """
        :param docs: list of docs
        each doc is a list of tuples
        each tuple contain normalized term, term position
        :return:
        """
        for doc_id, doc_info in enumerate(docs):
            self.add_new_doc(doc_id, doc_info)

    def add_new_doc(self, doc_id, doc):
        for position, term in enumerate(doc):
            if term not in self.index:
                self.index[term] = dict()
            if doc_id not in self.index[term]:
                self.index[term][doc_id] = []
            self.index[term][doc_id].append(position)

    def delete_doc(self, doc_id):
        for term in self.index.keys():
            self.index[term].pop(doc_id, None)

    def show_posting_list(self, term):
        print(self.name + " positional index")
        for doc_id, positions in self.index[term].items():
            print(str(doc_id) + " : " + str(positions))


    # TODO: add load & save methods

    def show_position(self, term, doc_id):
        print(self.name + " positional index")
        print(self.index[term][doc_id])

# test
# pi = PositionalIndex("test",
#                      [[("hello", 0), ("world", 1), ("!", 2)], [("fuck", 0), ("you", 1)], [("hello", 0)]])
