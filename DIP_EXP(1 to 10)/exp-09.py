import numpy as np
import heapq
from collections import defaultdict
import matplotlib.pyplot as plt
from PIL import Image

# Load the grayscale image
image_path = 'cameraman.tif'
image = Image.open(image_path)
image_array = np.array(image)

# Flatten the image array
data = image_array.flatten()

# Run-Length Encoding (RLE)
def run_length_encode(data):
    encoded_data = []
    count = 1
    for i in range(1, len(data)):
        if data[i] == data[i - 1]:
            count += 1
        else:
            encoded_data.append((data[i - 1], count))
            count = 1
    encoded_data.append((data[-1], count))
    return encoded_data

encoded_rle = run_length_encode(data)

# Huffman Coding
def build_frequency_table(data):
    freq_table = defaultdict(int)
    for item in data:
        freq_table[item] += 1
    return freq_table

def build_huffman_tree(freq_table):
    heap = [[weight, [symbol, ""]] for symbol, weight in freq_table.items()]
    heapq.heapify(heap)
    while len(heap) > 1:
        lo = heapq.heappop(heap)
        hi = heapq.heappop(heap)
        for pair in lo[1:]:
            pair[1] = '0' + pair[1]
        for pair in hi[1:]:
            pair[1] = '1' + pair[1]
        heapq.heappush(heap, [lo[0] + hi[0]] + lo[1:] + hi[1:])
    return sorted(heapq.heappop(heap)[1:], key=lambda p: (len(p[-1]), p))

def huffman_encode(data, huff_tree):
    encoding_dict = dict(huff_tree)
    encoded_data = ''.join(encoding_dict[item] for item in data)
    return encoded_data

def huffman_decode(encoded_data, huff_tree):
    decoding_dict = {code: symbol for symbol, code in huff_tree}
    decoded_data = ""
    code = ""
    for bit in encoded_data:
        code += bit
        if code in decoding_dict:
            decoded_data += decoding_dict[code]
            code = ""
    return decoded_data

freq_table = build_frequency_table(data)
huff_tree = build_huffman_tree(freq_table)
encoded_huffman = huffman_encode(data, huff_tree)

# Calculate compression ratios
original_size = data.size
rle_size = len(encoded_rle) * (8 + 8)  # Symbol + Count
huffman_size = len(encoded_huffman)

rle_ratio = original_size / rle_size
huffman_ratio = original_size / huffman_size

# Print compression ratios
print("Original size:", original_size)
print("RLE size:", rle_size)
print("Huffman size:", huffman_size)
print("RLE compression ratio:", rle_ratio)
print("Huffman compression ratio:", huffman_ratio)

