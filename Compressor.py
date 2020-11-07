from struct import pack, unpack


class Compressor:
    @staticmethod
    def variable_byte_encode(number):
        bytes = []
        while True:
            # ignore 0b
            binary_num = bin(number % 128)[2:]
            bin_num_str = '0' * (7 - len(binary_num)) + str(binary_num)
            if number < 128:
                bytes.append(int('1' + bin_num_str, 2))
                return pack('%dB' % len(bytes), *bytes)
            else:
                bytes.append(int('0' + bin_num_str, 2))
            number //= 128

    def variable_byte_decode(self, bytestream):
        num = 0
        numbers = []
        for byte in unpack('%dB' % len(bytestream), bytestream):
            if byte < 128:
                num = 128 * num + byte
            else:
                num = 128 * num + (byte - 128)
                numbers.append(num)
                num = 0
        return numbers