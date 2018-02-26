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

    rouges_arr = []
    bleu_arr = []

    tags = ['bleu','f1_score','precision','recall']
    for x in range(len(writers)):
        writers[x] = tf.summary.FileWriter(logdir + "/test/" + tags[x])

    bleu = tf.Variable(0.0)
    rouge1 = tf.Variable(0.0)
    rouge2 = tf.Variable(0.0)
    rouge3 = tf.Variable(0.0)

    # tf.summary.scalar("bleu", bleu)
    # tf.summary.scalar("rouge1_f1-p-r", rouge1)
    # tf.summary.scalar("rouge2_f1-p-r", rouge2)
    # tf.summary.scalar("rouge3_f1-p-r", rouge3)

    # write_op = tf.summary.merge_all()
    # session = tf.InteractiveSession()
    # session.run(tf.global_variables_initializer())

    for x in range(len(testX) -490):
        pred_y = model.predict(sess, testX[x].tolist(), metadata['idx2word'])
        true_y = [metadata['idx2word'][i] for i in testY[x]]

        p_y = ' '.join(pred_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()
        t_y = ' '.join(true_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()

        b = bleuMetric(t_y.split(" "), p_y.split(" "))

        r = [rouge_n(p_y.split(" "), t_y.split(" "), 1),
             rouge_n(p_y.split(" "), t_y.split(" "), 2),
             rouge_n(p_y.split(" "), t_y.split(" "), 3)]

        rouges_arr.append(rouge(p_y.split(" "), t_y.split(" ")))
        bleu_arr.append(b)
        # r = [[random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()]]

        # summary = session.run(write_op, {bleu: b})
        # writers[0].add_summary(summary, x)

        # for i in range(3):
        #     summary = session.run(write_op, {rouge1: r[0][i], rouge2: r[1][i], rouge3: r[2][i]})
        #     writers[i + 1].add_summary(summary, x)
        #     writers[i + 1].flush()

        # model.bleu = tf.convert_to_tensor(b)

        prt('Predicted:\t{} | \tTrue:\t{}'.format(p_y, t_y))

    metric_text = "\n\nROUGE-1 f1: \t\t%f\nROUGE-1 recall: \t%f\nROUGE-1 precision: \t%f\nROUGE-2 f1: \t\t%f\nROUGE-2 recall: \t%f\nROUGE-2 precision: \t%f\nROUGE-L f1: \t\t%f\nROUGE-L recall: \t%f\nROUGE-L precision: \t%f\n\nBLEU: \t%f\n" % (np.average([x['rouge_1/f_score'] for x in rouges_arr]),
                        np.average([x['rouge_1/r_score'] for x in rouges_arr]),
                        np.average([x['rouge_1/p_score'] for x in rouges_arr]),
                        np.average([x['rouge_2/f_score'] for x in rouges_arr]),
                        np.average([x['rouge_2/r_score'] for x in rouges_arr]),
                        np.average([x['rouge_2/p_score'] for x in rouges_arr]),
                        np.average([x['rouge_l/f_score'] for x in rouges_arr]),
                        np.average([x['rouge_l/r_score'] for x in rouges_arr]),
                        np.average([x['rouge_l/p_score'] for x in rouges_arr]),
                        np.average(bleu_arr))

    prt(metric_text)

    sess = tf.InteractiveSession()
    summary_op = tf.summary.text('config/config', tf.convert_to_tensor(metric_text))
    summary_writer = tf.summary.FileWriter(logdir + "/test/", sess.graph)
    text = sess.run(summary_op)
    summary_writer.add_summary(text, 0)
    summary_writer.add_summary(text, 100)
    summary_writer.add_summary(text, 200)
    summary_writer.flush()
    summary_writer.close()
