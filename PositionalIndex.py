from compressor import Compressor


class PositionalIndex:
    def __init__(self, name, docs):
        self.index = dict()
        self.name = name
        self.build(docs)

    def build(self, docs):
        """
        :param docs: list of docs
        each doc is a list of normalized term
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
            self.index[term][doc_id].append(position + 2)

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

    def compress(self, compression_type='var_byte'):
        compressed_index = dict()
        for term in self.index.keys():
            compressed_index[term] = dict()
            for doc_id in self.index[term]:
                if compression_type == 'var_byte':
                    compressed_index[term][doc_id] = Compressor.variable_byte_encode(self.index[term][doc_id])
                else:
                    compressed_index[term][doc_id] = Compressor.gamma_encode(self.index[term][doc_id])
        return compressed_index

    def decompress(self, compresses_index, compression_type='var_byte'):
        self.index = dict()
        for term in compresses_index.keys():
            self.index[term] = dict()
            for doc_id in compresses_index[term]:
                if compression_type == 'var_byte':
                    self.index[term][doc_id] = Compressor.variable_byte_decode(compresses_index[term][doc_id])
                else:
                    self.index[term][doc_id] = Compressor.gamma_decode(compresses_index[term][doc_id])

# test
# pi = PositionalIndex("test",
#                      [["hello", "world", "!"], ["fuck", "you"], ["hello"]])
# print(pi.index)
# cpi = pi.compress("gamma")
# print(cpi)
# pi.decompress(cpi, "gamma")
# print(pi.index)