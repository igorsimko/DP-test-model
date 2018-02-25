import datetime
import nltk
from rouge import *
import tensorflow as tf
from numpy import random

EOS_EL = 2


def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))


def bleuMetric(reference, hypothesis):
    return nltk.translate.bleu(reference, hypothesis)


def get_eos_pos(arr, return_val):
    if arr.any():
        if not np.where(arr == EOS_EL) == False:
            return np.where(arr == EOS_EL)[0][0]
        else:
            return return_val


def test(sess, model, metadata, testX, testY, logdir):
   # tf.reset_default_graph()

    now = datetime.datetime.now()
    tag = now.strftime("%Y%m%d-%H%M%S")
    writers = [None, None, None, None]

    tags = ['bleu','f1_score','precision','recall']
    for x in range(len(writers)):
        writers[x] = tf.summary.FileWriter(logdir + "/test/" + tags[x])

    bleu = tf.Variable(0.0)
    rouge1 = tf.Variable(0.0)
    rouge2 = tf.Variable(0.0)
    rouge3 = tf.Variable(0.0)

    tf.summary.scalar("bleu", bleu)
    tf.summary.scalar("rouge1_f1-p-r", rouge1)
    tf.summary.scalar("rouge2_f1-p-r", rouge2)
    tf.summary.scalar("rouge3_f1-p-r", rouge3)

    write_op = tf.summary.merge_all()
    # session = tf.InteractiveSession()
    sess.run(tf.global_variables_initializer())

    for x in range(len(testX)):
        pred_y = model.predict(sess, testX[x].tolist(), metadata['idx2word'])
        true_y = [metadata['idx2word'][i] for i in testY[x]]

        p_y = ' '.join(pred_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()
        t_y = ' '.join(true_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()

        b = bleuMetric(t_y.split(" "), p_y.split(" "))

        r = [rouge_n(p_y.split(" "), t_y.split(" "), 1),
             rouge_n(p_y.split(" "), t_y.split(" "), 2),
             rouge_n(p_y.split(" "), t_y.split(" "), 3)]

        # r = [[random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()]]

        summary = sess.run(write_op, {bleu: b})
        writers[0].add_summary(summary, x)

        for i in range(3):
            summary = sess.run(write_op, {rouge1: r[0][i], rouge2: r[1][i], rouge3: r[2][i]})
            writers[i + 1].add_summary(summary, x)
            writers[i + 1].flush()

        # model.bleu = tf.convert_to_tensor(b)

        prt('Predicted:\t{} | \tTrue:\t{}'.format(p_y, t_y))
