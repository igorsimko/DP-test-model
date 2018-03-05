import os
import pickle
import matplotlib.pyplot as plt
import plotly.plotly as py

from os import path
from nltk.corpus import wordnet

import pymysql
import pandas as pd

file_path = 'data/mysql'
pickle_file_name = "mysql_data.pkl"
ratio_int = 0.7

conn = pymysql.connect("localhost", "root", "root", "wiki",charset="utf8mb4",use_unicode=True)
cursor = conn.cursor()

cursor.execute('select version();')
cursor.execute('SET CHARACTER SET utf8;')

query1 = '''select convert(c.cl_to using utf8mb4) as cat, c.cl_from from wiki.categorylinks c limit 10;'''
query = '''
    select * from (
select
		p.page_id,
		convert(p.page_title using utf8mb4) as page_title,
		convert(t.old_text using utf8mb4) as txt,
		convert(c.cl_to using utf8mb4) as cat
    from wiki.page p
		join wiki.revision r on p.page_latest = r.rev_id
		join wiki.text t on t.old_id = r.rev_text_id
		join wiki.categorylinks c on c.cl_from = p.page_id
) as t
where 1=1
    and t.txt not like '%#REDIRECT%'
    and t.cat not like '%#All%pages%'
    and t.cat not like '%#All%articles%'
    and t.cat not like '%Disambiguation_pages%'
    and t.cat not like '%Articles%lacking%%'
	and t.cat not like '%Articles%needing%cleanup%'
	and t.cat not like '%Articles%from%'
	and t.cat not like '%Wiki%'
    and t.cat not like '%Articles%'
    and t.cat not like '%births%'
    and t.cat not like '%deaths%'
    and t.cat not like '%BC%'
	and t.cat not like '%Redirects%'
    and t.cat not like '%All%articles%'
	and t.cat not like '%stubs'
	and t.cat not like '%needing%'
	and t.cat not like '%List%'
	and t.cat not like '%people%'
    and t.cat not regexp '[[:digit:]]'
    and length(t.txt) < 30000
'''
def pickle_data(article_data):
    with open(path.join(file_path, pickle_file_name), 'wb') as fp:
        pickle.dump(article_data, fp, 2)

def unpickle_articles():
    with open(path.join(file_path, pickle_file_name), 'rb') as fp:
        article_data = pickle.load(fp)

    return article_data

def get_head_count(x):
    if len(x) % 2 == 0:
        return int(len(x)*ratio_int)
    else:
        return int(len(x)*ratio_int) + 1

def get_last_count(x):
    return int(len(x) * (1-ratio_int))

def contains_only_english_words(tokenized_sentence):
    contains = True

    for word in tokenized_sentence:
        if not wordnet.synsets(word):
            # Not an English Word
            contains = False
    if contains: return None

    return ' '.join(tokenized_sentence)

df = None
if os.path.exists(file_path + "/" + pickle_file_name):
    df = unpickle_articles()
else:
    df = pd.read_sql_query(query, conn)
    pickle_data(df)


df_temp = df.drop_duplicates(subset=['page_title'], keep='last').groupby('cat').filter(lambda x: len(x) > 4 and len(x) <= 15 )

print("Groups count: " + str(len(df_temp.groupby('cat').groups)))
df_test = df_temp.groupby('cat').apply(lambda x: x.tail(get_last_count(x)))
df_train = df_temp.groupby('cat').apply(lambda x: x.head(get_head_count(x)))

# df = df.drop_duplicates(subset=['page_title'], keep='last').groupby('cat').head(5).reset_index().sample(7000)
# print(len(df_train))
# print(len(df_test))
plt.hist(df_temp.groupby('cat').size().values)
plt.title("Početnosť kategórií")
plt.xlabel("Počet článkov v kategórii")
plt.ylabel("Frekvencia výskytu")
fig = plt.gcf()
plt.savefig('histogram.png')

with open('txtwiki/train-%d_test-%d.json' % (len(df_train), len(df_test)), 'w') as f:
    f.write(df_train.append(df_test).to_json(orient='records'))