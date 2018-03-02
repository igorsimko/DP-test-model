import datetime
import nltk
from rouge import *
import os
import tensorflow as tf
import numpy as np
from tensorflow.contrib.tensorboard.plugins import projector

EOS_EL = 2


def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))

def prt_out(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))
    return "[%s] - %s" % (datetime.datetime.now(), str)

def bleuMetric(reference, hypothesis):
    return nltk.translate.bleu(reference, hypothesis)


def get_eos_pos(arr, return_val):
    if arr.any():
        if not np.where(arr == EOS_EL) == False:
            return np.where(arr == EOS_EL)[0][0]
        else:
            return return_val


def test(sess, model, metadata, testX, testY, logdir, embedding):
   # tf.reset_default_graph()
    now = datetime.datetime.now()
    tag = now.strftime("%Y%m%d-%H%M%S")
    test_path = './test_logs/test-' + tag
    os.makedirs(test_path, 0o755)

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

    text_out = []
    for x in range(len(testX)):
        pred_y = model.predict(sess, testX[x].tolist(), metadata['idx2word'], embedding)
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
        text_out.append(prt_out('Predicted:\t{} | \tTrue:\t{}'.format(p_y, t_y)))

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

    text_file = open(test_path + "/metric.log", "w")
    text_file.write(metric_text)
    text_file.close()

    text_file = open(test_path + "/output_text.log", "w")
    text_file.write('\n'.join(text_out))
    text_file.close()

    #
    # sess = tf.InteractiveSession()
    # summary_op = tf.summary.text('config/config', tf.convert_to_tensor(metric_text))
    # summary_writer = tf.summary.FileWriter(logdir + "/test/", sess.graph)
    # text = sess.run(summary_op)
    # summary_writer.add_summary(text, 0)
    # summary_writer.add_summary(text, 100)
    # summary_writer.add_summary(text, 200)
    # summary_writer.flush()
    # summary_writer.close()





def visualize(model, output_path, tag):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), 50))

    with open(os.path.join(output_path,meta_file), 'wb') as file_metadata:
        for i, word in enumerate(model.wv.index2word):
            placeholder[i] = model[word]
            # temporary solution for https://github.com/tensorflow/tensorflow/issues/9094
            if word == '':
                print("Emply Line, should replaced by any thing else, or will cause a bug of tensorboard")
                file_metadata.write("{0}".format('<Empty Line>').encode('utf-8') + b'\n')
            else:
                file_metadata.write("{0}".format(word).encode('utf-8') + b'\n')

    # define the model without training
    sess = tf.InteractiveSession()

    embedding = tf.Variable(placeholder, trainable = False, name = 'w2x_metadata')
    tf.global_variables_initializer().run()

    saver = tf.train.Saver()
    writer = tf.summary.FileWriter(tag, sess.graph)

    # adding into projector
    config = projector.ProjectorConfig()
    embed = config.embeddings.add()
    embed.tensor_name = 'w2x_metadata'
    embed.metadata_path = meta_file

    # Specify the width and height of a single thumbnail.
    projector.visualize_embeddings(writer, config)
    saver.save(sess, os.path.join(tag,'w2x_metadata.ckpt'))
    # print('Run `tensorboard --logdir={0}` to run visualize result on tensorboard'.format(output_path))
