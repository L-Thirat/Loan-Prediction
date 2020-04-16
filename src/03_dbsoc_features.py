from dsproject import dsproject
from preprocess import *

dsp = dsproject()
db_soc = dsp.read_table('DB_SOC', 'preprocess', index_col=0)
db_soc['FileDate'] = parse_dates(db_soc['FileDate'], format='%Y%m%d')
db_soc = db_soc[(db_soc['FileDate'] >= dsp.min_date_dbsoc) & (db_soc['FileDate'] < dsp.max_date_dbsoc)]
db_soc = db_soc[['HASH_AR_ID', 'FileDate', 'PortFolio', 
                 'ProductName', 'OSPRINCIPLE', 'OSAMT', 
                 'BILLCycle', 'Range_day', 'Card Type New']]

len(db_soc.drop_duplicates(subset=['HASH_AR_ID', 'FileDate']).index)
dsp.write_table(db_soc, 'dbsoc', 'feature', 
                ds_type=['id', 'date', 'category', 'category', 'numeric', 
                         'numeric', 'category', 'category', 'category'])

print(len(db_soc.index))
print(db_soc['FileDate'].min())
print(db_soc['FileDate'].max())