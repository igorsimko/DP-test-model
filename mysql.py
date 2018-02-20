import pymysql
import pandas as pd

conn = pymysql.connect("localhost", "root", "root", "wiki",charset="utf8mb4",use_unicode=True)
cursor = conn.cursor()

cursor.execute('select version();')
cursor.execute('SET CHARACTER SET utf8;')

query1 = '''select convert(c.cl_to using utf8mb4) as cat from wiki.categorylinks c limit 10;'''
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
    and t.cat not regexp '[[:digit:]]'
    and length(t.txt) < 5000
limit 5000
'''



df = pd.read_sql_query(query, conn)
print(len(df))
with open('mysql_select_out.json', 'w') as f:
    f.write(df.to_json( orient='records'))
result = cursor.fetchone()
print(result)