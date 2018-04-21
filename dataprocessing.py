import json
from os import path
import metrics

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from collections import namedtuple

import unicodedata
import sentence_similarity
import utils

from gensim.models import Word2Vec
from gensim.summarization import keywords
from nltk.tokenize import word_tokenize as tokenize
import nltk
import itertools
import numpy as np
import cloudpickle as pickle
import pandas as pd
from nltk.corpus import wordnet
import re
from join import parse_text
from rouge import rouge

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ .'
VOCAB_SIZE = 50000

split_ratio = 0.8
embedding = 15
just_test_changes = False

CM_KEYWORDS = 'keywords'
CM_PARSE = 'n-grams'
CM_GET_MOST_SIMILIAR = 'cos_sim'
CM_GET_MOST_SIMILIAR_WITH_PARSE = 'cos_sim_w_n-grams'
CM_KEYWORDS_PARSE = 'keywords_n-grams'

UNK = '<UNK>'
GO = '<GO>'
EOS = '<EOS>'
PAD = '<PAD>'

file_path = 'data'

file_name = 'train-4557_test-1212.json'
file_name = 'train-5185_test-1671.json'
file_name = 'train-8562_test-2807.json'
file_name = 'train-6441_test-5475.json'
file_name = 'train-223_test-70.json'
# file_name = 'train-7608_test-4247.json'

global_separator = int(file_name.split("_")[0].split('-')[1])

limit = {
    'max_descriptions': 150,
    'min_descriptions': 0,
    'max_headings': 8,
    'min_headings': 0,
}

def get_limits():
    return limit

def load_raw_data(filename):
    with open(filename, 'r', encoding='utf8') as fp:
        raw_data = pd.DataFrame(json.load(fp))

    print('Loaded {:,} articles from {}'.format(len(raw_data), filename))
    return raw_data


def tokenize_sentence(sentence):
    if not sentence:
        return []

    if isinstance(sentence, bytes):
        sentence = sentence.decode('utf8')
    if isinstance(sentence, list):
        return sentence
    return ' '.join(list(tokenize(sentence)))


def contains_only_english_words(tokenized_sentence, enable=True):
    if not enable:
        return True
    contains = True

    for word in tokenized_sentence:
        if not wordnet.synsets(word):
            # Not an English Word
            contains = False

    return contains


def cut_limits(x, limit):
    return ' '.join(x.split(" ")[:limit])


# English Word

def article_is_complete(article):
    if ('category' not in article) or ('parsed_text' not in article):
        return False
    if (article['category'] is None) or (article['parsed_text'] is None):
        return False
    if (len(article['category']) == 0) or (len(article['parsed_text']) == 0):
        return False

    return True


def tokenize_articles(raw_data, test_arr, split_separator=0):
    headings, descriptions, test_categories = [], [], []
    num_articles = len(raw_data)
    nltk.download('punkt')

    separator_counter = 0
    new_separator = 0
    for i, row in raw_data.iterrows():
        if i == global_separator or (split_separator != 0 and i == split_separator):
            new_separator = separator_counter
        if article_is_complete(row) and contains_only_english_words(row['category'].split(" "), enable=False):
            separator_counter = separator_counter + 1

            if len(test_arr) > 0 and 'page_id' in row: test_categories.append(
                test_arr.loc[test_arr['page_id'] == row['page_id']]['categories'].values)
            headings.append(tokenize_sentence(row['category']))
            descriptions.append(tokenize_sentence(row['parsed_text']))
        if i % 1 == 0:
            print('Tokenized {:,} / {:,} articles'.format(separator_counter, num_articles))

    print("New separator idx: " + str(new_separator if new_separator != 0 else split_separator))
    return (headings, descriptions, test_categories, new_separator)


def filter(line, whitelist):
    return ''.join([ch for ch in line if ch in whitelist])


def filter_length(headings, descriptions):
    if len(headings) != len(descriptions):
        raise Exception('Number of headings does not match number of descriptions!')

    filtered_headings, filtered_descriptions = [], []

    for i in range(0, len(headings)):
        heading_length = len(headings[i].split(' '))
        description_length = len(descriptions[i].split(' '))

        if description_length >= limit['min_descriptions'] and description_length <= limit['max_descriptions']:
            if heading_length >= limit['min_headings'] and heading_length <= limit['max_headings']:
                filtered_headings.append(headings[i])
                filtered_descriptions.append(descriptions[i])

    print('Length of filtered headings: {:,}'.format(len(filtered_headings)))
    print('Length of filtered descriptions: {:,}'.format(len(filtered_descriptions)))
    return (filtered_headings, filtered_descriptions)


def index_data(tokenized_sentences, vocab_size):
    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    print('Vocab length: {:,}'.format(len(vocab)))

    idx2word = [GO] + [UNK] + [EOS] + [PAD] + [x[0] for x in vocab]
    word2idx = dict([(w, i) for i, w in enumerate(idx2word)])

    return (idx2word, word2idx, freq_dist)


def pad_seq(seq, lookup, max_length, old_flag=False):
    indices = []

    for word in seq:
        if word in lookup:
            indices.append(lookup[word])
        else:
            indices.append(lookup[UNK])

    if old_flag:
        return indices + [lookup[EOS]] + [lookup[PAD]] * (max_length - len(seq) - 1)
    else:
        pad_arr = []
        while len(pad_arr) < (max_length - len(seq)):
            for i in indices:
                pad_arr.append(i)
                if len(pad_arr) + len(indices) == max_length:
                    break
        # return indices + [lookup[EOS]] + [lookup[PAD]]*(max_length - len(seq) - 1)
        return indices + pad_arr


def group_by_category(category_pd, category_to_categories, raw_data, methods):
    ret_df = pd.DataFrame(columns=['category', 'parsed_text'])
    cm = {}

    # groups data
    for i, group in enumerate(list(category_pd.groups.keys())):
        print(group + " | %d/%d" % (i, len(category_pd)))

        for id in list(category_pd.get_group(group)['page_id']):
            append_to_dict(category_to_categories, group, raw_data.loc[raw_data['page_id'] == id]['category'].values)

        # sim_cat_words = parse_text(' '.join(
        #     keywords(filter(category_pd.get_group(group)['parsed_text'].values[0], whitelist=WHITELIST)).split(' ')[
        #     :limit['max_descriptions']])).split(' ')

        if group not in cm:
            cm[group] = {}
            cm[group][CM_KEYWORDS] = {}
            cm[group][CM_PARSE] = {}
            cm[group][CM_GET_MOST_SIMILIAR] = {}
            cm[group][CM_GET_MOST_SIMILIAR_WITH_PARSE] = {}
            cm[group][CM_KEYWORDS_PARSE] = {}

        sim_cat_words_arr = []

        if len(category_pd.get_group(group)['parsed_text'].values) != 0:

            if CM_KEYWORDS in methods: cm[group][CM_KEYWORDS] = keywords(' '.join(category_pd.get_group(group)['parsed_text'].values)).split('\n')
            if CM_PARSE in methods: cm[group][CM_PARSE] = parse_text(' '.join(category_pd.get_group(group)['parsed_text'].values)).split(' ')
            if CM_GET_MOST_SIMILIAR in methods: cm[group][CM_GET_MOST_SIMILIAR] = sentence_similarity.get_most_similiar_words_by_category(category_pd.get_group(group)['parsed_text'].values, treshold=0.6)
            if CM_GET_MOST_SIMILIAR_WITH_PARSE in methods: cm[group][CM_GET_MOST_SIMILIAR_WITH_PARSE] = sentence_similarity.get_most_similiar_words_by_category([parse_text(x) for x in category_pd.get_group(group)['parsed_text'].values])
            if CM_KEYWORDS_PARSE in methods: cm[group][CM_KEYWORDS_PARSE] = sentence_similarity.gramatic_keyword(category_pd.get_group(group)['parsed_text'].values)

            if CM_KEYWORDS in methods: sim_cat_words_arr.append(cm[group][CM_KEYWORDS][:limit['max_descriptions']])
            if CM_PARSE in methods: sim_cat_words_arr.append(cm[group][CM_PARSE][:limit['max_descriptions']])
            if CM_KEYWORDS_PARSE in methods: sim_cat_words_arr.append(cm[group][CM_KEYWORDS_PARSE][:limit['max_descriptions']])
            if CM_GET_MOST_SIMILIAR in methods: sim_cat_words_arr.append(cm[group][CM_GET_MOST_SIMILIAR][:limit['max_descriptions']])
            if CM_GET_MOST_SIMILIAR_WITH_PARSE in methods: sim_cat_words_arr.append(cm[group][CM_GET_MOST_SIMILIAR_WITH_PARSE][:limit['max_descriptions']])


        if sim_cat_words_arr and len(sim_cat_words_arr) > 0:
            for sim_cat_words in sim_cat_words_arr:
                ret_df = ret_df.append(
                    pd.DataFrame([[group, ' '.join(sim_cat_words)]], columns=['category', 'parsed_text']))

    return ret_df, cm

def cm_get_arr_alias(metric, cm, arr ):
    if metric == 'ROUGE':
        return arr[cm].ROUGE
    elif metric == 'SIMILARITY':
        return arr[cm].SIMILARITY
    elif metric == 'BLEU':
        return arr[cm].BLEU

def eval_methods(x, y):
    compare_methods = {CM_KEYWORDS: metrics.Metrics(),
                       CM_PARSE: metrics.Metrics(),
                       CM_GET_MOST_SIMILIAR: metrics.Metrics(),
                       CM_GET_MOST_SIMILIAR_WITH_PARSE: metrics.Metrics(),
                       CM_KEYWORDS_PARSE: metrics.Metrics()}
    for yi in y:
        if yi in x:
            for cm in list(compare_methods.keys()):
                true_y =  y[yi][cm]
                pred_y =  x[yi][cm]

                if len(true_y) == 0 or len(pred_y) == 0:
                    b = 0
                    ss = 0
                    rr = 0
                else:
                    b = utils.bleuMetric(true_y, pred_y)
                    ss = sentence_similarity.sentence_similarity(' '.join(pred_y), ' '.join(true_y))
                    rr = rouge(pred_y, true_y)['rouge_1/f_score']

                compare_methods[cm].ROUGE.append(rr)
                compare_methods[cm].SIMILARITY.append(ss)
                compare_methods[cm].BLEU.append(b)

                # print("%s\t- ROUGE: %f | BLEU: %f | SIMILARITY: %f" %(cm, rr, b, ss))

    print("\n--- AVG metrics ---\n")

    avg_eval = {CM_KEYWORDS: metrics.Metrics(),
                       CM_PARSE: metrics.Metrics(),
                       CM_GET_MOST_SIMILIAR: metrics.Metrics(),
                       CM_GET_MOST_SIMILIAR_WITH_PARSE: metrics.Metrics(),
                       CM_KEYWORDS_PARSE: metrics.Metrics()
    }

    for cm in list(compare_methods.keys()):
        avg_bleu = np.average(compare_methods[cm].BLEU)
        avg_rouge = np.average(compare_methods[cm].ROUGE)
        avg_sim = np.average(compare_methods[cm].SIMILARITY)

        avg_eval[cm].ROUGE.append(avg_rouge)
        avg_eval[cm].SIMILARITY.append(avg_sim)
        avg_eval[cm].BLEU.append(avg_bleu)

        print("AVG (BLEU) - %s\t - %f" %(cm, avg_bleu))
        print("AVG (ROUGE) - %s\t - %f" %(cm, avg_rouge))
        print("AVG (SIM) - %s\t - %f" %(cm, avg_sim))


    fig, ax = plt.subplots()
    index = np.arange(0, 5 * 2, 2)
    bar_width = 0.35

    data_bleu = ax.bar(index, tuple([x[1].ROUGE[0] for x in list(avg_eval.items())]), bar_width,
                alpha=1, color='blue',
                label='ROUGE')
    data_rouge = ax.bar(index + bar_width, tuple([x[1].BLEU[0] for x in list(avg_eval.items())]), bar_width,
                alpha=1, color='red',
                label='BLEU')
    data_sim = ax.bar(index + bar_width + bar_width, tuple([x[1].SIMILARITY[0] for x in list(avg_eval.items())]), bar_width,
                alpha=1, color='green',
                label='SIMILARITY')

    ax.set_xlabel('Metódy')
    ax.set_ylabel('Skóre')
    ax.set_title('Porovnanie metód pre zjednotenie dokumentov')
    ax.set_xticks(index + bar_width / 2)
    for tick in ax.get_xticklabels():
        tick.set_rotation(25)
    ax.set_xticklabels((CM_KEYWORDS, CM_PARSE, CM_GET_MOST_SIMILIAR, CM_GET_MOST_SIMILIAR_WITH_PARSE, CM_KEYWORDS_PARSE))
    ax.legend()

    fig.tight_layout()
    plt.savefig('eval-methods-%s.png' % file_name)

    print('\n')
    print("\n--- AVG metrics ---\n")


    return compare_methods


def zero_pad(tokenized_headings, tokenized_descriptions, word2idx, only_desc=False):
    data_length = len(tokenized_descriptions)
    max_desc_len = len(max(tokenized_descriptions, key=len)) + 1
    max_head_len = len(max(tokenized_headings, key=len)) + 1 if not only_desc else 0
    idx_descriptions = np.zeros([data_length, max_desc_len], dtype=np.int32)
    idx_headings = np.zeros([data_length, max_head_len], dtype=np.int32)
    # idx_descriptions = []
    # idx_headings = []

    for i in range(data_length):
        description_indices = pad_seq(tokenized_descriptions[i], word2idx, max_desc_len, old_flag=True)
        heading_indices = pad_seq(tokenized_headings[i], word2idx, max_head_len, old_flag=True) if not only_desc else []

        idx_descriptions[i] = np.array(description_indices)
        idx_headings[i] = np.array(heading_indices)

    return (idx_headings, idx_descriptions)


def remove_underscore(data, where=None):
    if where == None:
        return data
    data[where].apply(lambda x: x.replace('_', " "))


def append_to_dict(arr, key, value):
    if key not in arr:
        arr[key] = []

    if isinstance(value, list):
        for i in value:
            arr[key].append(i)
    else:
        arr[key].append(value)


def process_data():
    # load data from file
    filename = path.join(file_path, file_name)
    raw_data = load_raw_data(filename)
    raw_data['category'] = raw_data['category'].apply(lambda x: x.replace('_', " "))
    raw_data['category'] = raw_data['category'].apply(lambda x: x.lower())

    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('{(.*?)}', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('\|(.*?)=(.*?)', " ", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('}*', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('{*', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('[\n]+', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('== External links ==\s*([^\n\r]*)', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.split('Category:')[0])

    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('=', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('.', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('\\', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('b\'', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('/', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.replace('*', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('\s+', " ", x))

    utils.prt("Start parsing n-grams")
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: str(unicodedata.normalize('NFKD', x).encode('ascii', 'ignore')))

    # Average document length (words)
    # utils.prt( np.average([len(x.split(' ')) for x in raw_data['parsed_text'].values]))
    # utils.prt( np.average([len(x) for x in raw_data['parsed_text'].apply(lambda x: sentence_similarity.gramatic_keyword([x])).values]))

    raw_data['category'] = raw_data['category'].apply(lambda x: ' '.join(x.split(' ')[:limit['max_headings']]))
    # raw_data['parsed_text'] = raw_data['parsed_text'].apply(
    #     lambda x: ' '.join(x.split(' ')[:limit['max_descriptions']]))

    # group_by_id = raw_data.groupby('page_id')
    test_arr = pd.DataFrame(columns=['page_id', 'parsed_text', 'categories'])
    # for i in group_by_id.groups.keys():
    #     print("%s : %d" %(i, len(group_by_id.get_group(i)['category'].values)))
    #     test_arr = test_arr.append(pd.DataFrame([[i, group_by_id.get_group(i)['parsed_text'].head(1), group_by_id.get_group(i)['category'].values]], columns=['page_id', 'parsed_text', 'categories']))

    # with open('data/test_data_%d.json' % global_separator , 'w') as f:
    #     f.write(test_arr.to_json(orient='records'))

    utils.prt('Spliting test/train DF')

    model = Word2Vec([x.split(' ') for x in raw_data['parsed_text'].values], min_count=1, size=embedding)
    model.save('model.bin')
    # RANDOM GROUPS FOR TESTING PURPOSE

    # random_groups = []
    # groups_by_fletter = raw_data.head(global_separator).groupby(raw_data.head(global_separator)['category'].str.get(0))
    # for group in list(groups_by_fletter.groups):
    #     for cat in groups_by_fletter.get_group(group).sample(5)['category'].values:
    #         random_groups.append(cat)

    # train data
    category_to_categories = {}
    train_df = raw_data.head(global_separator).groupby('category')
    sim_train_df, cm_train = group_by_category(train_df, category_to_categories, raw_data, methods=[CM_PARSE])
    # test data
    test_df = raw_data.tail(len(raw_data) - global_separator).groupby('category')

    split_ratio_index = len(test_df) * (1 - split_ratio)
    test_df = pd.DataFrame([j for i in [g[1].values for g in list(test_df)[:int(split_ratio_index)]] for j in i], columns=['category', 'page_id', 'page_title', 'parsed_text']).groupby('category')
    utils.prt("Length whole: %d" % (len(raw_data.groupby('category'))))
    utils.prt("Length train: %d | Length test: %d" % (len(train_df), len(test_df)))
    sim_test_df, cm_test = group_by_category(test_df, category_to_categories, raw_data, methods=[CM_KEYWORDS_PARSE])

    #eval_methods(cm_train, cm_test)
    raw_data = sim_train_df.append(sim_test_df)

    raw_data.to_json('parsed.json', orient='records')


    headings, descriptions, test_categories, new_separator = tokenize_articles(raw_data, test_arr, split_separator=len(sim_train_df))

    # keep only whitelisted characters and articles satisfying the length limits
    # headings = [filter(heading, WHITELIST) for heading in headings]
    # descriptions = [filter(sentence, WHITELIST) for sentence in descriptions]
    # headings, descriptions = filter_length(headings, descriptions)


    # convert list of sentences into list of list of words
    word_tokenized_headings = [word_list.split(' ') for word_list in headings]
    word_tokenized_descriptions = [word_list.split(' ') for word_list in descriptions]

    # indexing
    idx2word, word2idx, freq_dist = index_data(word_tokenized_headings + word_tokenized_descriptions, VOCAB_SIZE)

    # save as numpy array and do zero padding
    idx_headings, idx_descriptions = zero_pad(word_tokenized_headings, word_tokenized_descriptions, word2idx)

    # check percentage of unks
    new_idx_h = []
    new_idx_d = []
    new_test_arr = []

    for index, i in enumerate(idx_headings):
        if 1 not in i:
            new_idx_h.append(i)
            new_idx_d.append(idx_descriptions[index])
            if len(test_categories) > 0: new_test_arr.append(test_categories[index])

    idx_headings = new_idx_h
    idx_descriptions = new_idx_d
    # unk_percentage = calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)
    # print (calculate_unk_percentage(idx_headings, idx_descriptions, word2idx))

    tmp_pickle = None

    if just_test_changes:
        tmp_pickle = unpickle_articles()

    if tmp_pickle != None and just_test_changes:
        article_data = {
            'word2idx': tmp_pickle['word2idx'],
            'idx2word': tmp_pickle['idx2word'],
            'limit': tmp_pickle['limit'],
            'freq_dist': tmp_pickle['freq_dist'],
            'idx_headings': idx_headings,
            'idx_descriptions': idx_descriptions,
            # 'test_categories': new_test_arr[new_separator:]
            'test_categories': tmp_pickle['test_categories']
        }
    else:
        article_data = {
            'word2idx': word2idx,
            'idx2word': idx2word,
            'limit': limit,
            'freq_dist': freq_dist,
            'idx_headings': idx_headings,
            'idx_descriptions': idx_descriptions,
            # 'test_categories': new_test_arr[new_separator:]
            'test_categories': category_to_categories
        }

    pickle_data(article_data)

    return (idx_headings, idx_descriptions)


def pickle_data(article_data):
    with open(path.join(file_path, 'article_data.pkl'), 'wb') as fp:
        pickle.dump(article_data, fp, 2)


def unpickle_articles():
    with open(path.join(file_path, 'article_data.pkl'), 'rb') as fp:
        article_data = pickle.load(fp)

    return article_data


def calculate_unk_percentage(idx_headings, idx_descriptions, word2idx):
    num_unk = (idx_headings == word2idx[UNK]).sum() + (idx_descriptions == word2idx[UNK]).sum()
    num_words = (idx_headings > word2idx[UNK]).sum() + (idx_descriptions > word2idx[UNK]).sum()

    return (num_unk / num_words) * 100


def main():
    process_data()


if __name__ == '__main__':
    main()
