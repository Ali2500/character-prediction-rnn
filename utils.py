import numpy as np
from definitions import N_CHARS


def str_to_one_hot(string):
    int_str = [ord(x) for x in string]
    one_hot = np.zeros(shape=(len(string), N_CHARS), dtype=np.uint8)

    for i in range(len(int_str)):
        if int_str[i] == 10:  # new line character
            one_hot[i, 0] = 1
        elif 32 <= int_str[i] <= 126:
            one_hot[i, int_str[i] - 31] = 1
        else:
            raise ValueError("[ERROR] Found invalid character %d. String: %s" % (int_str[i], string))

    return one_hot


def str_to_idx_arr(string):
    int_str = np.array([ord(x) for x in string]) - 31
    int_str[int_str == -21] = 0
    return int_str.astype(np.uint8)


def one_hot_to_str(one_hot):
    if one_hot.shape[1] != N_CHARS:
        raise ValueError("Unexpected 2nd dim length of one-hot array: %d, expected: %d" % (one_hot.shape[1], N_CHARS))
    return idx_arr_to_str(np.argmax(one_hot, axis=1))


def idx_arr_to_str(idx_arr):
    chr_arr = idx_arr + 31
    chr_arr[chr_arr == 31] = 10
    return ''.join(chr(x) for x in chr_arr)
