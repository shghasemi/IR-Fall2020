import pickle
from math import floor, log2


class Compressor:
    @staticmethod
    def convert_binary_str_to_bytes(bin_str):
        n = int(bin_str, 2)
        b = bytearray()
        while n:
            b.append(n & 0xff)
            n >>= 8
        return bytes(b[::-1])

    @staticmethod
    def convert_bytes_to_binary_str(byte_arr):
        result = ""
        first_byte_flag = True
        for b in byte_arr:
            byte_str = str(bin(b)[2:])
            if not first_byte_flag:
                result += '0' * (8 - len(byte_str))
            else:
                first_byte_flag = False
            result += byte_str
        return result

    @staticmethod
    def get_gaps_list(numbers):
        if len(numbers) == 0:
            return []
        gaps = [numbers[0]]
        for i in range(len(numbers) - 1):
            gaps.append(numbers[i + 1] - numbers[i])
        return gaps

    @staticmethod
    def get_numbers_list(gaps):
        if len(gaps) == 0:
            return []
        numbers = [gaps[0]]
        for i in range(len(gaps) - 1):
            numbers.append(numbers[i] + gaps[i + 1])
        return numbers

    @staticmethod
    def variable_byte_encode(numbers):
        gaps = Compressor.get_gaps_list(numbers)
        code_str = ""
        for n in gaps:
            code_str += Compressor.variable_byte_encode_number(n)
        return Compressor.convert_binary_str_to_bytes(code_str)

    @staticmethod
    def variable_byte_encode_number(number):
        number_str = ""
        byte_str_list = []
        while True:
            binary_num = bin(number % 128)[2:]
            byte_str_list.append('0' * (8 - len(binary_num)) + str(binary_num))
            if number < 128:
                break
            number //= 128
        low_byte = list(byte_str_list[0])
        low_byte[0] = '1'
        byte_str_list[0] = "".join(low_byte)
        for i in range(len(byte_str_list) - 1, -1, -1):
            number_str += byte_str_list[i]
        return number_str

    @staticmethod
    def variable_byte_decode(code):
        n = 0
        gaps = []
        for byte in code:
            n = 128 * n + byte
            if byte // 128 > 0:
                n -= 128
                gaps.append(n)
                n = 0
        return Compressor.get_numbers_list(gaps)

    @staticmethod
    def gamma_encode(numbers):
        gaps = Compressor.get_gaps_list(numbers)
        code_str = ""
        for n in gaps:
            code_str += Compressor.gamma_encode_number(n)
        return Compressor.convert_binary_str_to_bytes(code_str)

    @staticmethod
    def gamma_encode_number(number):
        code = ""
        for _ in range(floor(log2(number))):
            code += '1'
        binary_num = bin(number)[2:]
        code += "0"
        for i in range(1, len(binary_num)):
            code += binary_num[i]
        return code

    @staticmethod
    def gamma_decode(codes):
        codes_str = Compressor.convert_bytes_to_binary_str(codes)
        codes_str_len = len(codes_str)
        pos = 0
        gaps = []
        # for i in range(4):
        while pos < codes_str_len:
            counter = 0
            while codes_str[pos + counter] == '1':
                counter += 1
            n = 1
            for i in range(pos + counter + 1, pos + 2 * counter + 1):
                n *= 2
                if codes_str[i] == '1':
                    n += 1
            pos += 2 * counter + 1
            gaps.append(n)
        return Compressor.get_numbers_list(gaps)
