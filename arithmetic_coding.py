# arithmetic_coding.py

import io
from fractions import Fraction

def arithmetic_coding(text):
    frequency = {char: text.count(char) for char in set(text)}
    cumulative_freq = dict()
    sum_freq = 0
    for char in sorted(frequency):
        cumulative_freq[char] = sum_freq
        sum_freq += frequency[char]
    high = 1
    low = 0
    for char in text:
        range_size = high - low
        high = low + range_size * (cumulative_freq[char] + frequency[char]) / len(text)
        low = low + range_size * cumulative_freq[char] / len(text)
    return (high + low) / 2

import math

def arithmetic_compression(text):
    code = arithmetic_coding(text)
    # Convert the float to a Fraction object
    code_fraction = Fraction(code).limit_denominator()
    # Compute the number of bits needed to represent the float
    num_bits = math.ceil(math.log2(code_fraction.denominator))
    return '1' * num_bits
