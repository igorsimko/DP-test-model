import data_utils
from utils import *
from dataprocessing import *
from rouge import *
import seq2seq_wrapper
import tensorflow as tf
import os.path
import sys
import json

# tensorboard --logdir=E:\git\DP-test-model\tf_logs --port 6006


conf_number = sys.argv[1]

config = []

if os.path.exists("config.json"):
    with open('config.json', 'r') as f:
        config = json.load(f)
    actual_config = config['config_number_' + conf_number]
else:
    # default hyperparams
    actual_config = {
        'batch_size' : 32,
        'emb_dim' : 15,
        'num_layers' : 2,
        'epochs' : 10,
        'num_units' : 50,
        'learning_rate' : 0.001,
        'split_idx': 4557,
        'ckpt' : "ckpt/",
        'logdir' : "./tf_logs"
    }

learning_rate = actual_config['learning_rate']
epochs = actual_config['epochs']
num_units = actual_config['num_units']
num_layers = actual_config['num_layers']
emb_dim = actual_config['emb_dim']
epochs = actual_config['epochs']
batch_size = actual_config['batch_size']
split_idx = actual_config['split_idx']
ckpt = actual_config['ckpt']
logdir = actual_config['logdir']

logdir = '%s/%s' % (logdir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

metadata = unpickle_articles()  # data.load_data(PATH='datasets/twitter/')
idx_a, idx_q = metadata['idx_headings'], metadata['idx_descriptions']

(trainX, trainY), (testX, testY), (validX, validY) = data_utils.split_data(idx_q, idx_a)

# parameters
xseq_len = len(trainX[0])
yseq_len = len(trainY[0])
xvocab_size = len(metadata['idx2word'])
yvocab_size = xvocab_size

model = Word2Vec.load('model.bin')

embedding_matrix = np.zeros((len(metadata['idx2word']), emb_dim))
for i in range(len(metadata['idx2word'])):
    if metadata['idx2word'][i] not in model.wv:
        embedding_vector = np.zeros((emb_dim))
    else:
        embedding_vector = model.wv[metadata['idx2word'][i]]
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector
embedding_matrix = np.array(embedding_matrix, dtype=np.float64)

visualize(model, "tf_logs", logdir)
model = seq2seq_wrapper.Seq2Seq(xseq_len=xseq_len,
                                yseq_len=yseq_len,
                                xvocab_size=xvocab_size,
                                yvocab_size=yvocab_size,
                                ckpt_path=ckpt,
                                emb_dim=emb_dim,
                                num_layers=num_layers,
                                num_units=num_units,
                                epochs=epochs,
                                emb_size=emb_dim
                                )

# val_batch_gen = data_utils.rand_batch_gen(validX, validY, batch_size)
# train_batch_gen = data_utils.rand_batch_gen(trainX, trainY, batch_size)
# test_batch_gen = data_utils.rand_batch_gen(testX, testY, batch_size)

train_batch_gen = data_utils.rand_batch_gen(idx_q[:split_idx], idx_a[:split_idx], batch_size)
test_batch_gen = data_utils.rand_batch_gen(idx_q[split_idx:], idx_a[split_idx:], batch_size)

sess = None

def restore_session():
    with tf.Session() as sess:
        # Initialize v1 since the saver will not.
        saver = tf.train.Saver()
        saver.restore(sess, ckpt + "model.ckpt")

        test(sess, model, metadata, testX, testY, logdir, embedding_matrix)

if os.path.exists(ckpt + "checkpoint"):
    restore_session()
else:
    # saved_embeddings = np.append(np.zeros((4, emb_dim)), embedding_matrix, axis=0)
    model.fit(trainX, trainY, log_dir=logdir, val_data=(testX, testY), batch_size=batch_size, embedding=embedding_matrix)
    restore_session()



