import data_utils
from utils import *
from dataprocessing import *
import seq2seq_wrapper
import tensorflow as tf
from datetime import datetime

# tensorboard --logdir=run1:/tf_logs/ --port 6006

metadata = unpickle_articles()  # data.load_data(PATH='datasets/twitter/')
idx_a, idx_q = metadata['idx_headings'], metadata['idx_descriptions']

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_data(idx_q, idx_a)


# hyperparams
batch_size = 32
emb_dim = 15
num_layers = 2
epochs = 1
num_units = 50

now = datetime.now()
#now.strftime("%Y%m%d-%H%M%S")
logdir = "/tf_logs/"  + ""

# parameters
xseq_len = len(trainX[0])
yseq_len = len(trainY[0])
# xseq_len = 30
# yseq_len = 6
xvocab_size = len(metadata['idx2word'])
yvocab_size = xvocab_size

model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path='ckpt/',
                                emb_dim=emb_dim,
                                num_layers=num_layers,
                                num_units=num_units,
                                epochs=epochs,
                                emb_size=emb_dim
                                )

val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)

# sess = model.train(train_batch_gen, val_batch_gen)
sess = model.fit(trainX, trainY, log_dir=logdir, val_data=(testX, testY), batch_size=batch_size)


for x in range(len(testX)):
    pred_y = model.predict(sess, testX[x].tolist(), metadata['idx2word'])
    true_y = [metadata['idx2word'][i] for i in testY[x]]

    p_y = ' '.join(pred_y).replace("<PAD>", "").replace("<EOS>","")
    t_y = ' '.join(true_y).replace("<PAD>", "").replace("<EOS>","")

    model.bleu = bleu(t_y.split(" "), p_y.split(" "))
    prt('BLEU:{} | \tPredicted:\t{} | \tTrue:\t{}'.format(model.bleu, p_y, t_y))

    # for ii, oi in zip(input_.T, output):
    #     q = data_utils.decode(sequence=ii, lookup=metadata['idx2word'], separator=' ')
    #     decoded = data_utils.decode(sequence=oi, lookup=metadata['idx2word'], separator=' ').split(' ')
    #     if decoded.count('unk') == 0:
    #         if decoded not in replies:
    #             model.bleu = bleu(data_utils.decode(sequence=testY[x], lookup=metadata['idx2word'], separator=' '),
    #                               decoded)
    #             print('description : [{0}]; category : [{1}]'.format(q, ' '.join(decoded)))
    #             replies.append(decoded)
