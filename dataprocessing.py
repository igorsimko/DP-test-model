import json
from os import path

import unicodedata
from nltk.tokenize import word_tokenize as tokenize
import nltk
import itertools
import numpy as np
import cloudpickle as pickle
import pandas as pd
import re
from join import parse_text

WHITELIST = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ '
VOCAB_SIZE = 20000

UNK = '<UNK>'
GO = '<GO>'
EOS = '<EOS>'
PAD = '<PAD>'

file_path = 'data'
file_name = 'train_10K_5K-len.json'
file_name = '4K_unique.json'
file_name = '5K_cat_unique.json'
file_name = '7K_not_unique.json'


limit = {
    'max_descriptions' : 500,
    'min_descriptions' : 0,
    'max_headings' : 100,
    'min_headings' : 0,
}

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

def article_is_complete(article):
    if ('category' not in article) or ('parsed_text' not in article):
        return False
    if (article['category'] is None) or (article['parsed_text'] is None):
        return False

    return True

def tokenize_articles(raw_data):
    headings, descriptions = [], []
    num_articles = len(raw_data)
    nltk.download('punkt')

    for i, row in raw_data.iterrows():
        if article_is_complete(row):
            headings.append(tokenize_sentence(row['category']))
            descriptions.append(tokenize_sentence(row['parsed_text']))
        if i % 1 == 0:
            print('Tokenized {:,} / {:,} articles'.format(i, num_articles))

    return (headings, descriptions)

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


def zero_pad(tokenized_headings, tokenized_descriptions, word2idx):
    data_length = len(tokenized_descriptions)
    max_desc_len = len(max(tokenized_descriptions, key=len)) + 1
    max_head_len = len(max(tokenized_headings, key=len)) + 1
    idx_descriptions = np.zeros([data_length, max_desc_len], dtype=np.int32)
    idx_headings = np.zeros([data_length, max_head_len], dtype=np.int32)
    # idx_descriptions = []
    # idx_headings = []

    for i in range(data_length):
        description_indices = pad_seq(tokenized_descriptions[i], word2idx, max_desc_len, old_flag=True)
        heading_indices = pad_seq(tokenized_headings[i], word2idx, max_head_len, old_flag=True)

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

    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: x[:limit['max_descriptions']])

    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: parse_text(x))
    raw_data['parsed_text'] = raw_data['parsed_text'].apply(lambda x: unicodedata.normalize('NFKD', x).encode('ascii','ignore'))

    raw_data.to_json('parsed.json', orient='records')
    headings, descriptions = tokenize_articles(raw_data)

    #keep only whitelisted characters and articles satisfying the length limits
    headings = [filter(heading, WHITELIST) for heading in headings]
    descriptions = [filter(sentence, WHITELIST) for sentence in descriptions]
    headings, descriptions = filter_length(headings, descriptions)

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

    for index, i in enumerate(idx_headings):
        if 1 not in i:
            new_idx_h.append(i)
            new_idx_d.append(idx_descriptions[index])

    idx_headings = new_idx_h
    idx_descriptions = new_idx_d
    # unk_percentage = calculate_unk_percentage(idx_headings, idx_descriptions, word2idx)
    # print (calculate_unk_percentage(idx_headings, idx_descriptions, word2idx))

    article_data = {
        'word2idx' : word2idx,
        'idx2word': idx2word,
        'limit': limit,
        'freq_dist': freq_dist,
        'idx_headings' : idx_headings,
        'idx_descriptions': idx_descriptions
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
