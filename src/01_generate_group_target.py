from dsproject import dsproject
from preprocess import *

dsp = dsproject()

db_soc = dsp.read_table('DB_SOC', 'preprocess', index_col=0, dtype={'HASH_IP_ID':str, 'HASH_AR_ID':str} )
db_soc = db_soc.loc[db_soc['Group'].isin([1,3,5])]
db_soc = db_soc.loc[db_soc['OSAMT'] > 100]

db_soc['FileDate'] = parse_dates(db_soc['FileDate'], format='%Y%m%d')
db_soc = db_soc[(db_soc['FileDate'] >= dsp.min_date_dbsoc) & (db_soc['FileDate'] < dsp.max_date_dbsoc)]

#Generate unique filedate table
unique_filedate = db_soc[['BILLCycle', 'FileDate']].drop_duplicates()

unique_filedate['Next_FileDate'] = unique_filedate.groupby('BILLCycle')[['BILLCycle','FileDate']].shift(-1)
unique_filedate['Previous_FileDate'] = unique_filedate.groupby('BILLCycle')[['BILLCycle','FileDate']].shift(1)
db_soc = db_soc.merge(unique_filedate, on=['BILLCycle', 'FileDate'])

unique_filedate = db_soc[['HASH_AR_ID', 'FileDate', 'Previous_FileDate', 'Next_FileDate']].drop_duplicates()
dsp.write_table(unique_filedate, 'unique_filedate', 'feature')

# add HASH_IP_ID
df_id = dsp.read_data('EDW_LPM_X_AR_X_CIS', usecols=['HASH_IP_ID','HASH_AR_ID'], dtype={'HASH_IP_ID':str,'HASH_AR_ID':str})
df = db_soc.merge(df_id, on=['HASH_AR_ID'], how ='inner')
print(df.head())
print("Merge coverage")
print(1.0*len(df.index)/len(db_soc.index))

df = df[['HASH_AR_ID', 'HASH_IP_ID', 'HASH_LPM_CST_ID', 'FileDate', 'Previous_FileDate', 'Next_FileDate', 'Group', 'OSAMT']]
dsp.write_table(df, 'group_target', 'feature', ds_type=['id', 'id', 'id', 'date', 'date', 'date', 'category', 'numeric'])
