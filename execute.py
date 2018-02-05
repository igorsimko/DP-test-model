import data_utils
from utils import *
from dataprocessing import *
import seq2seq_wrapper
import tensorflow as tf

metadata = unpickle_articles() #data.load_data(PATH='datasets/twitter/')
idx_a, idx_q = metadata['idx_headings'], metadata['idx_descriptions']

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_data(idx_q, idx_a)

# parameters 
xseq_len = len(trainX[0])
yseq_len = len(trainY[0])
batch_size = 32
xvocab_size = len(metadata['idx2word'])
yvocab_size = xvocab_size
emb_dim = 50


model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                               yseq_len=yseq_len,
                               xvocab_size=xvocab_size,
                               yvocab_size=yvocab_size,
                               ckpt_path='ckpt/',
                               emb_dim=emb_dim,
                               num_layers=3,
                               epochs=1000
                               )


val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 5)

sess = model.train(train_batch_gen, val_batch_gen)

for x in range(len(testX)):
    input_ = test_batch_gen.__next__()[0]
    output = model.predict(sess, input_)

    replies = []
    for ii, oi in zip(input_.T, output):
        q = data_utils.decode(sequence=ii, lookup=metadata['idx2word'], separator=' ')
        decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2word'], separator=' ').split(' ')
        if decoded.count('unk') == 0:
            if decoded not in replies:
                model.bleu = bleu(data_utils.decode(sequence=testY[x], lookup=metadata['idx2word'], separator=' '), decoded)
                print('description : [{0}]; category : [{1}]'.format(q, ' '.join(decoded)))
                replies.append(decoded)
