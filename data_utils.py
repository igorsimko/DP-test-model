import numpy as np
from random import sample

def split_data(x, y, ratio=[0.8, 0.1, 0.1]):
    data_length = len(x)
    lengths = [int(data_length * item) if int(data_length * item) > 0 else 1 for item in ratio]

    x_train, y_train = x[:lengths[0]], y[:lengths[0]]
    x_test, y_test = x[lengths[0]:lengths[0]+lengths[1]], y[lengths[0]:lengths[0]+lengths[1]]
    x_valid, y_valid = x[-lengths[-1]:], y[-lengths[-1]:]

    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)


def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield np.array(x)[sample_idx].T, np.array(y)[sample_idx].T


def split_by_idx(x, y, idx):
    return (x[:idx], y[:idx]), (x[idx:], y[idx:])

def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])
