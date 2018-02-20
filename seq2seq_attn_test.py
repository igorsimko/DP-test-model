import data_utils
from utils import *
from dataprocessing import *
import seq2seq_wrapper
import tensorflow as tf
from seq2seq_attn import *

# tensorboard --logdir=run1:/tf_logs/ --port 6006

metadata = unpickle_articles()  # data.load_data(PATH='datasets/twitter/')
idx_a, idx_q = metadata['idx_headings'], metadata['idx_descriptions']
word2idx, idx2word = metadata['word2idx'], metadata['idx2word']

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_data(idx_q, idx_a)

# parameters
xseq_len = len(trainX[0])
yseq_len = len(trainY[0])
batch_size = 32
xvocab_size = len(metadata['idx2word'])
yvocab_size = xvocab_size
emb_dim = 15
embedding = tf.nn.embedding_lookup(trainX, emb_dim)

# model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
#                                 yseq_len=yseq_len,
#                                 xvocab_size=xvocab_size,
#                                 yvocab_size=yvocab_size,
#                                 ckpt_path='ckpt/',
#                                 emb_dim=emb_dim,
#                                 num_layers=2,
#                                 num_units=32,
#                                 epochs=1,
#                                 emb_size=emb_dim,
#                                 embedding=embedding
#                                 )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, 32)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, 5)

model = Seq2Seq(
    rnn_size = 50,
    n_layers = 2,
    X_word2idx = word2idx,
    encoder_embedding_dim = 15,
    Y_word2idx = word2idx,
    decoder_embedding_dim = 15,
    batch=32
)
model.fit(trainX, trainY, val_data=(testX, testY), n_epoch=10, batch_size=batch_size)
# model.train(train_batch_gen, val_batch_gen, xseq_len, yseq_len)# model.infer('common', X_idx2char, Y_idx2char)
# model.infer('apple', idx2word, idx2word)
# model.infer('zhedong', X_idx2char, Y_idx2char)


for x in range(len(testX)):
    pred_y = model.infer(testX[x].tolist(), idx2word, idx2word)

    # pred_y = model.predict(testX[x].tolist(), metadata['idx2word'])
    true_y = [metadata['idx2word'][i] for i in testY[x]]

    prt('Predicted:\t{}\nTrue:\t{}'.format(' '.join(pred_y), ' '.join(true_y)))
    model.bleu = bleu(true_y, pred_y)

if __name__ == '__main__':
    main()
