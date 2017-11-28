import numpy as np
from random import sample

'''
 split data into train (70%), test (15%) and valid(15%)
    return tuple( (trainX, trainY), (testX,testY), (validX,validY) )

'''
def split_data(x, y, ratio=[0.8, 0.1, 0.1]):
    data_length = len(x)
    lengths = [int(data_length * item) if int(data_length * item) > 0 else 1 for item in ratio]

    x_train, y_train = x[:lengths[0]], y[:lengths[0]]
    x_test, y_test = x[lengths[0]:lengths[0]+lengths[1]], y[lengths[0]:lengths[0]+lengths[1]]
    x_valid, y_valid = x[-lengths[-1]:], y[-lengths[-1]:]

    return (x_train, y_train), (x_test, y_test), (x_valid, y_valid)


'''
 generate batches from dataset
    yield (x_gen, y_gen)

    TODO : fix needed

'''
def batch_gen(x, y, batch_size):
    # infinite while
    while True:
        for i in range(0, len(x), batch_size):
            if (i+1)*batch_size < len(x):
                yield x[i : (i+1)*batch_size ].T, y[i : (i+1)*batch_size ].T

'''
 generate batches, by random sampling a bunch of items
    yield (x_gen, y_gen)

'''
def rand_batch_gen(x, y, batch_size):
    while True:
        sample_idx = sample(list(np.arange(len(x))), batch_size)
        yield x[sample_idx].T, y[sample_idx].T

#'''
# convert indices of alphabets into a string (word)
#    return str(word)
#
#'''
#def decode_word(alpha_seq, idx2alpha):
#    return ''.join([ idx2alpha[alpha] for alpha in alpha_seq if alpha ])
#
#
#'''
# convert indices of phonemes into list of phonemes (as string)
#    return str(phoneme_list)
#
#'''
#def decode_phonemes(pho_seq, idx2pho):
#    return ' '.join( [ idx2pho[pho] for pho in pho_seq if pho ])


'''
 a generic decode function 
    inputs : sequence, lookup

'''
def decode(sequence, lookup, separator=''): # 0 used for padding, is ignored
    return separator.join([ lookup[element] for element in sequence if element ])
