import json
from os import path

import unicodedata
import sentence_similarity
import utils

from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize as tokenize
import nltk
import itertools
import numpy as np
import cloudpickle as pickle
import pandas as pd
from nltk.corpus import wordnet
import re
from join import parse_text

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
VOCAB_SIZE = 50000

UNK = '<UNK>'
GO = '<GO>'
EOS = '<EOS>'
PAD = '<PAD>'

file_path = 'data'

global_separator = 5185

file_name = 'train_10K_5K-len.json'
file_name = '4K_unique.json'
file_name = '5K_cat_unique.json'
file_name = '7K_not_unique.json'
file_name = 'train-4557_test-1212.json'
file_name = 'train-5185_test-1671.json'
# file_name = 'train-61_test-19.json'

limit = {
    'max_descriptions' : 100,
    'min_descriptions' : 0,
    'max_headings' : 5,
    'min_headings' : 0,
}

def loadGloveModel(gloveFile):
    print("Loading Glove Model")
    f = open(gloveFile,'r', encoding='utf8')
    model = {}
    for line in f:
        splitLine = line.split()
        word = splitLine[0]
        embedding = np.array([float(val) for val in splitLine[1:]])
        model[word] = embedding
    print("Done.",len(model)," words loaded!")
    return model

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

def contains_only_english_words(tokenized_sentence):
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
    if (len(article['category']) == 0 ) or (len(article['parsed_text']) == 0):
        return False

    return True

def tokenize_articles(raw_data, test_arr):
    headings, descriptions, test_categories = [], [], []
    num_articles = len(raw_data)
    nltk.download('punkt')

    separator_counter = 0
    new_separator = 0
    for i, row in raw_data.iterrows():
        if i == global_separator:
            new_separator = separator_counter
        if article_is_complete(row) and contains_only_english_words(row['category'].split(" ")):
            separator_counter = separator_counter + 1

            test_categories.append(test_arr.loc[test_arr['page_id']==row['page_id']]['categories'].values)
            headings.append(tokenize_sentence(row['category']))
            descriptions.append(tokenize_sentence(row['parsed_text']))
        if i % 1 == 0:
            print('Tokenized {:,} / {:,} articles'.format(i, num_articles))

    print("New separator idx: " + str(new_separator))
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

    print ('Length of filtered headings: {:,}'.format(len(filtered_headings)))
    print ('Length of filtered descriptions: {:,}'.format(len(filtered_descriptions)))
    return (filtered_headings, filtered_descriptions)

def index_data(tokenized_sentences, vocab_size):

    freq_dist = nltk.FreqDist(itertools.chain(*tokenized_sentences))
    vocab = freq_dist.most_common(vocab_size)
    print ('Vocab length: {:,}'.format(len(vocab)))

    idx2word = [GO] + [UNK] + [EOS] + [PAD] +[x[0] for x in vocab]
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
        return indices + [lookup[EOS]] + [lookup[PAD]]*(max_length - len(seq) - 1)
    else:
        pad_arr = []
        while len(pad_arr) < (max_length - len(seq)):
            for i in indices:
                pad_arr.append(i)
                if len(pad_arr) + len(indices) == max_length:
                    break
        # return indices + [lookup[EOS]] + [lookup[PAD]]*(max_length - len(seq) - 1)
        return indices + pad_arr


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


def process_data():

    #load data from file
    filename = path.join(file_path, file_name)
    raw_data = load_raw_data(filename)
    raw_data['category'] = raw_data['category'].apply(lambda x: x.replace('_', " "))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('{(.*?)}', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('\|(.*?)=(.*?)', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('}*', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('{*', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('[\n]+', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: re.sub('== External links ==\s*([^\n\r]*)', "", x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x.split('Category:')[0])
    utils.prt("Start parsing n-grams")
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore'))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: parse_text(x))

    raw_data['category'] = raw_data['category'].apply(lambda x: ' '.join(x.split(' ')[:limit['max_headings']]))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: ' '.join(x.split(' ')[:limit['max_descriptions']]))


    group_by_id = raw_data.groupby('page_id')
    test_arr = pd.DataFrame(columns=['page_id', 'parsed_text', 'categories'])
    for i in group_by_id.groups.keys():
        test_arr = test_arr.append(pd.DataFrame([[i, group_by_id.get_group(i)['parsed_text'].head(1), group_by_id.get_group(i)['category'].values]], columns=['page_id', 'parsed_text', 'categories']))

    # with open('data/test_data_%d.json' % global_separator , 'w') as f:
    #     f.write(test_arr.to_json(orient='records'))

    # utils.prt('Spliting test/train DF')
    # train_df = raw_data.head(global_separator).groupby('category')
    # test_df = raw_data.tail(len(raw_data) - global_separator).groupby('category')
    # sim_train_df = pd.DataFrame(columns=['category', 'parsed_text'])
    # sim_test_df = pd.DataFrame(columns=['category', 'parsed_text'])
    #
    # # train data
    # for i, group in enumerate(list(train_df.groups.keys())):
    #     print(group + " | %d/%d" % (i, len(train_df)))
    #     sim_cat_words = sentence_similarity.get_most_similiar_words_by_category(
    #         train_df.get_group(group)['parsed_text'].values, treshold=0.8)[:limit['max_descriptions']]
    #     if len (sim_cat_words) > 0:
    #         sim_train_df = sim_train_df.append(pd.DataFrame([[group, ' '.join(sim_cat_words)]], columns=['category', 'parsed_text', 'categories']))
    #
    # # test data
    # for i, group in enumerate(list(test_df.groups.keys())):
    #     print(group + " | %d/%d" % (i, len(train_df)))
    #     sim_cat_words = sentence_similarity.get_most_similiar_words_by_category(
    #         test_df.get_group(group)['parsed_text'].values, treshold=0.8)[:limit['max_descriptions']]
    #     if len(sim_cat_words) > 0:
    #         # categories = list(raw_data[raw_data['page_id'].isin(raw_data.groupby('category').get_group(group)['page_id'].values)].groupby('category').groups.keys())
    #         sim_test_df = sim_test_df.append(pd.DataFrame([[group, ' '.join(sim_cat_words)]], columns=['category', 'parsed_text']))
    #
    #
    # raw_data = sim_train_df.append(sim_test_df)

    # raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x[:limit['max_descriptions']])
    # raw_data['category'] = raw_data['category'].apply(lambda x: parse_text(x))


    # raw_data.to_json('parsed.json', orient='records')

    # model = loadGloveModel("glove/glove.6B.50d.txt")

    headings, descriptions, test_categories, new_separator = tokenize_articles(raw_data, test_arr)

    #keep only whitelisted characters and articles satisfying the length limits
    # headings = [filter(heading, WHITELIST) for heading in headings]
    # descriptions = [filter(sentence, WHITELIST) for sentence in descriptions]
    # headings, descriptions = filter_length(headings, descriptions)

    # np.savetxt('dp_target.txt', headings, fmt='%s')
    # np.savetxt('dp_target.txt', descriptions, fmt='%s')

    #convert list of sentences into list of list of words
    word_tokenized_headings = [word_list.split(' ') for word_list in headings]
    word_tokenized_descriptions = [word_list.split(' ') for word_list in descriptions]

    #indexing
    idx2word, word2idx, freq_dist = index_data(word_tokenized_headings + word_tokenized_descriptions, VOCAB_SIZE)

    #save as numpy array and do zero padding
    idx_headings, idx_descriptions = zero_pad(word_tokenized_headings, word_tokenized_descriptions, word2idx)

    #check percentage of unks
    new_idx_h = []
    new_idx_d = []
    new_test_arr = []

    for index, i in enumerate(idx_headings):
        if 1 not in i:
            new_idx_h.append(i)
            new_idx_d.append(idx_descriptions[index])
            if len(test_categories) > 0 : new_test_arr.append(test_categories[index])

    idx_headings = new_idx_h
    idx_descriptions = new_idx_d
    # unk_percentage = calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)
    # print (calculate_unk_percentage(idx_headings, idx_descriptions, word2idx))

    model = Word2Vec(word_tokenized_descriptions + word_tokenized_headings, min_count=1, size=15)
    model.save('model.bin')

    article_data = {
        'word2idx' : word2idx,
        'idx2word': idx2word,
        'limit': limit,
        'freq_dist': freq_dist,
        'idx_headings' : idx_headings,
        'idx_descriptions': idx_descriptions,
        'test_categories': new_test_arr[new_separator:]
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
