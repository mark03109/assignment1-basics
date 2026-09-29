# print(chr(0))
# print(1)
# a = 43
# s = "string"
# l: list = [1, 2, 3, 4]
# print(repr(chr(0)))
# print(repr(a))
# print(repr(s))
# print(repr(l))
# print(l)

# chr(0)
# print(chr(0))
# a = "this is a test" + chr(0) + "string"
# print(repr(a))
# s = "this is a test" + "string"
# print(repr(s))
# print("this is a test" + chr(0) + "string")
# print("this is a test" + "string")

# string_orig = "hello, 你好啊"
# string_encode = string_orig.encode("utf-8")
# string_encode_list = list(string_encode)
# print(type(string_orig))
# print(type(string_encode))
# print(type(string_encode[0]))
# print(type(string_encode_list))
# print(type(string_encode_list[0]))

# string_orig = "hello, 你好啊"
# string_encode_8 = string_orig.encode("utf-8")
# string_encode_16 = string_orig.encode("utf-16")
# print(string_encode_8.hex())
# print(string_encode_16.hex())

# def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
#     return "".join([bytes([b]).decode("utf-8") for b in bytestring])

# print(decode_utf8_bytes_to_str_wrong("hello".encode("utf-8")))
# s = "你好"
# print(decode_utf8_bytes_to_str_wrong(s.encode("utf-8")))

string_encode = bytes([200, 232])
print(string_encode.decode("utf-8"))