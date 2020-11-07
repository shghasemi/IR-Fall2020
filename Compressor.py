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
        for b in byte_arr:
            result += str(bin(b)[2:])
        return result

    @staticmethod
    def variable_byte_encode(number):
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
        return Compressor.convert_binary_str_to_bytes(number_str)

    @staticmethod
    def variable_byte_decode(code):
        n = 0
        numbers = []
        for byte in code:
            n = 128 * n + byte
            if byte // 128 > 0:
                n -= 128
                numbers.append(n)
                n = 0
        return numbers

    @staticmethod
    def gamma_encode(number):
        if number == 0:
            return '0'
        code = ""
        for _ in range(floor(log2(number))):
            code += '1'
        binary_num = bin(number)[2:]
        code += "0"
        for i in range(1, len(binary_num)):
            code += binary_num[i]
        return code

    @staticmethod
    def gamma_decode(code):
        counter = 0
        while code[counter] == '1':
            counter += 1
        if counter == 0:
            return 0
        n = 1
        for i in range(counter + 1, 2 * counter + 1):
            n *= 2
            if code[i] == '1':
                n += 1
        return n
