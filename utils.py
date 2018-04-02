import datetime
import nltk
import pandas as pd
from rouge import *
import os
import tensorflow as tf
import numpy as np
import textwrap
import sentence_similarity as sensim
from tensorflow.contrib.tensorboard.plugins import projector

EOS_EL = 2


def prt(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))

def prt_out(str):
    print("[%s] - %s" % (datetime.datetime.now(), str))
    return "[%s] - %s" % (datetime.datetime.now(), str)

def bleuMetric(reference, hypothesis, smoothing_function=None):
    return nltk.translate.bleu_score.sentence_bleu([reference], hypothesis, smoothing_function=smoothing_function)


def get_eos_pos(arr, return_val):
    if arr.any():
        if not np.where(arr == EOS_EL) == False:
            return np.where(arr == EOS_EL)[0][0]
        else:
            return return_val

def postprocess_predict(predicate):
    ret = []
    for word in predicate:
        if (len(ret) == 0) or (len(ret) > 0 and ret[len(ret) - 1] != word):
            ret.append(word)

    return ret


def test(sess, model, metadata, testX, testY, logdir, embedding, trace=False):
    now = datetime.datetime.now()
    tag = now.strftime("%Y%m%d-%H%M%S")
    test_path = './test_logs/test-' + tag
    os.makedirs(test_path, 0o755)

    writers = [None, None, None, None]

    rouges_arr = [None]*len(testY)
    bleu_arr = np.zeros(len(testY))
    sensim_arr = np.zeros(len(testY))

    tags = ['bleu','f1_score','precision','recall']
    for x in range(len(writers)):
        writers[x] = tf.summary.FileWriter(logdir + "/test/" + tags[x])

    text_out = []

    for x in range(len(testX)):
        x_for_test = testX[x]
        # remove unk tokens
        x_for_test = [x for x in x_for_test if x != 1]
        print(x_for_test)

        pred_y = model.predict(sess, x_for_test, metadata['idx2word'], embedding)
        pred_y = postprocess_predict(pred_y)

        true_y = [metadata['idx2word'][i] for i in testY[x]] if len(testY) > 0 else ""

        p_y = ' '.join(pred_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()
        t_y = ' '.join(true_y).replace("<PAD>", "").replace("<EOS>", "").rstrip().lstrip()

        b = bleuMetric(t_y.split(" "), p_y.split(" "))
        ss = sensim.sentence_similarity(p_y, t_y)
        rr = rouge(p_y.split(" "), t_y.split(" "))

        rouges_arr[x] = rr
        bleu_arr[x] = b
        sensim_arr[x] = ss

        if metadata['test_categories']:
            t_y_metric_temp = []
            t_y_cat_temp = ""
            if t_y in metadata['test_categories']:
                categories = [j for i in metadata['test_categories'][t_y] for j in i]
                for category in set(categories):
                    metric_ratio = categories.count(t_y) / len(metadata['test_categories'][t_y])
                    temp_b = bleuMetric(category.split(" "), p_y.split(" ")) *  metric_ratio
                    temp_ss = sensim.sentence_similarity(p_y, category) * metric_ratio
                    temp_rr = rouge(p_y.split(" "), category.split(" "))

                    if temp_b > b:
                        b = temp_b
                        t_y_metric_temp.append("bleu")
                        t_y_cat_temp = category

                    if temp_ss > ss:
                        ss = temp_ss
                        t_y_metric_temp.append("sim")
                        t_y_cat_temp = category

                    if temp_rr['rouge_1/f_score'] * metric_ratio > rr['rouge_1/f_score']:
                        rr = temp_rr
                        t_y_metric_temp.append("(rouge)")
                        t_y_cat_temp = category

                if len(t_y_metric_temp) != 0:
                    t_y = ','.join(set(t_y_metric_temp)) + " | " + t_y_cat_temp + " (instead of) " + t_y



        # r = [rouge_n(p_y.split(" "), t_y.split(" "), 1),
        #      rouge_n(p_y.split(" "), t_y.split(" "), 2),
        #      rouge_n(p_y.split(" "), t_y.split(" "), 3)]

        # r = [[random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()], [random.rand(),random.rand(),random.rand()]]

        # summary = session.run(write_op, {bleu: b})
        # writers[0].add_summary(summary, x)

        # for i in range(3):
        #     summary = session.run(write_op, {rouge1: r[0][i], rouge2: r[1][i], rouge3: r[2][i]})
        #     writers[i + 1].add_summary(summary, x)
        #     writers[i + 1].flush()

        # model.bleu = tf.convert_to_tensor(b)
        if ss != 0:
            if trace:
                text_sample = ' '.join([metadata['idx2word'][i] for i in x_for_test])[:280] + "..."
                text_out.append(prt_out('Sample {}\n\nMachine generated category: {}\nReal category: {}\n'.format(x + 1, p_y, t_y) + '\nSimilarity: %f\n\n' % ss + textwrap.fill('Text sample: {}\n\n'.format(text_sample), 78)))

        if trace != True:
            text_out.append(prt_out('Predicted:\t{} | \tTrue:\t{}'.format(p_y, t_y)))

    metric_text = "===== All predics =====\n\nROUGE-1 f1: \t\t%f\n" \
                  "ROUGE-1 recall: \t%f\n" \
                  "ROUGE-1 precision: \t%f\n" \
                  "ROUGE-2 f1: \t\t%f\n" \
                  "ROUGE-2 recall: \t%f\n" \
                  "ROUGE-2 precision: \t%f\n" \
                  "ROUGE-L f1: \t\t%f\n" \
                  "ROUGE-L recall: \t%f\n" \
                  "ROUGE-L precision: \t%f\n\n" \
                  "BLEU: \t%f\n\n" \
                  "Sentence similarity: \t%f\n" % (np.average([x['rouge_1/f_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_1/r_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_1/p_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_2/f_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_2/r_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_2/p_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_l/f_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_l/r_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average([x['rouge_l/p_score'] for x in rouges_arr]),
                                                                                                                                                                                                                                                                                np.average(bleu_arr),
                                                                                                                                                                                                                                                                                np.average(sensim_arr))

    prt(metric_text)

    metric_text = "===== At least one good average =====\n\nROUGE-1 f1: \t\t%f\nROUGE-1 recall: \t%f\nROUGE-1 precision: \t%f\nROUGE-2 f1: \t\t%f\nROUGE-2 recall: \t%f\nROUGE-2 precision: \t%f\nROUGE-L f1: \t\t%f\nROUGE-L recall: \t%f\nROUGE-L precision: \t%f\n\nBLEU: \t%f\n\nSentence similarity: \t%f\n" % (
        np.average(([x['rouge_1/f_score'] for x in rouges_arr if x['rouge_1/f_score'] != 0])),
        np.average(([x['rouge_1/r_score'] for x in rouges_arr if x['rouge_1/r_score'] != 0])),
        np.average(([x['rouge_1/p_score'] for x in rouges_arr if x['rouge_1/p_score'] != 0])),
        np.average(([x['rouge_2/f_score'] for x in rouges_arr if x['rouge_2/f_score'] != 0])),
        np.average(([x['rouge_2/r_score'] for x in rouges_arr if x['rouge_2/r_score'] != 0])),
        np.average(([x['rouge_2/p_score'] for x in rouges_arr if x['rouge_2/p_score'] != 0])),
        np.average(([x['rouge_l/f_score'] for x in rouges_arr if x['rouge_l/f_score'] != 0])),
        np.average(([x['rouge_l/r_score'] for x in rouges_arr if x['rouge_l/r_score'] != 0])),
        np.average(([x['rouge_l/p_score'] for x in rouges_arr if x['rouge_l/p_score'] != 0])),
        np.average([x for x in bleu_arr if x != 0]),
        np.average([x for x in sensim_arr if x != 0 ]))

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





def visualize(model, output_path, tag, emb):
    meta_file = "w2x_metadata.tsv"
    placeholder = np.zeros((len(model.wv.index2word), emb))

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
