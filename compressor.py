import pickle
from struct import pack, unpack
from sys import getsizeof
from bitarray import bitarray as _bitarray


class bitarray(_bitarray):
    def __hash__(self):
        return id(self)


class Compressor:
    def __init__(self, pos_index, name):
        self.index = pos_index
        self.name = name
        self.compressed = {}

    def encode_gamma(self, number):
        # TODO change bitarray
        # binary_n = format(number, 'b')
        # binary_offset = binary_n[1::]
        # unary_length = bitarray(True for i in range(len(binary_offset))) + bitarray([False])
        # return bitarray(unary_length), bitarray(binary_offset)
        number += 1
        # to binary
        if number == 0:
            return [0]
        binary = []
        while number >= 1:
            binary.append(number % 2)
            number = int(number / 2)
        length = len(binary) - 1
        output_binary = [1 for _ in range(length)]
        output_binary.append(0)
        for i in range(len(binary) - 1):
            output_binary.append(binary[len(binary) - 2 - i])
        bytes = list(reversed(output_binary))
        return pack('%dB' % len(bytes), *bytes)

    def encode_vbytecode(self, number):
        bytes = []
        # while True:
        #     bytes.insert(0, number % 128)
        #     if number < 128:
        #         break
        #     number //= 128
        # bytes[-1] += 128
        # bytes = []
        while True:
            b = bin(number % 128)[2:]
            number = number // 128
            if number == 0:
                b = '1' + '0' * (8 - len(b) - 1) + b
                bytes.append(int(b, 2))
                break
            else:
                b = '0' * (8 - len(b)) + b
            bytes.append(int(b, 2))
        return pack('%dB' % len(bytes), *bytes)

    def decode_gamma(self, bytestream):
        pass

    def decode_vbytecode(self, bytestream):
        num = 0
        numbers = []
        bytestream = unpack('%dB' % len(bytestream), bytestream)
        for byte in bytestream:
            if byte < 128:
                num = 128 * num + byte
            else:
                num = 128 * num + (byte - 128)
                numbers.append(num)
                num = 0
        return numbers

    def encode(self, number, name):
        if name == 'gamma':
            return self.encode_gamma(number)
        else:
            return self.encode_vbytecode(number)

    def compress(self, compress_offsets=False):

        for term in self.index:
            prev_doc_id = 0
            self.compressed[term] = {}
            for doc_id in self.index[term]:
                gap = doc_id - prev_doc_id
                prev_doc_id = doc_id
                encoded_doc_id = self.encode(gap, self.name)

                # encode offsets
                if compress_offsets:
                    offsets = self.index[term][doc_id][:]
                    offsets.insert(0, 0)
                    encoded_offsets = [self.encode(s - t, self.name) for t, s in zip(offsets, offsets[1:])]
                else:
                    encoded_offsets = self.index[term][doc_id]

                self.compressed[term][encoded_doc_id] = encoded_offsets

    def save(self):
        with open('index/' + self.name + '.pkl', 'wb') as f:
            pickle.dump(self.compressed, f, pickle.HIGHEST_PROTOCOL)

    def load(self):
        with open('index/' + self.name + '.pkl', 'rb') as f:
            self.compressed = pickle.load(f)

    def space_dif(self):
        print("Memory allocation before compression: ", getsizeof(pickle.dumps(self.index)), "bytes")
        print("Memory allocation after compression: ", getsizeof(pickle.dumps(self.compressed)), "bytes")
        print("Decreased: ", getsizeof(pickle.dumps(self.index)) - getsizeof(pickle.dumps(self.compressed)))
