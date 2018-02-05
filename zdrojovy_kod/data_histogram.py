import pymysql
import pymysql.cursors
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate

conn = pymysql.connect(host='localhost', user='root', password='root', db='wiki', charset='utf8',
                       cursorclass=pymysql.cursors.DictCursor)
a = conn.cursor()
sql = 'select cc.pid, cc.c, cc.cl from (\
	SELECT cl_from as pid, count(*) as c, convert(cl_to using utf8) as \'cl\' FROM wiki.categorylinks\
    group by cl\
	\
) as cc order by cc.c desc '
a.execute(sql)

batch = 10000
arr = a.fetchall()
stop_words = ['article', 'redirect', 'people', 'index', 'pages', 'stub', 'wiki', 'births', 'deaths', 'containing']

by_batches = {}
for i in range(0, a.rowcount, batch):
    batch_arr = arr[i:(i + batch)]
    stops_count = 0
    for k in batch_arr:
        for stopw in stop_words:
            if stopw in k['cl']:
                stops_count += 1
                break
    by_batches.update({str(i): stops_count})

print(by_batches)
x = by_batches.keys()
y = by_batches.values()


import matplotlib.ticker as plticker


plt.scatter(x, y)
#plt.xticks(np.arange(min(map(int,x)), max(map(int,x))+1, 10000.0))
plt.xticks(rotation=90)

plt.show()

