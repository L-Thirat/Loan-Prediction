import os
from dsproject import dsproject
from preprocess import *

dsp = dsproject()

f = open(os.path.join(dsp.data_directory, 'EDW_SUM_CC_AR_BHVR_SCOR_Header.txt'),'r')
f = f.read().replace('\r\n','')
header = f.split('|')

df0 = dsp.read_data('EDW_SUM_CC_AR_BHVR_SCOR')

df = delete_missing_columns(df0)  
df = delete_single_value_columns(df)

df['POSN_DT'] = parse_dates(df['POSN_DT'], format='%Y-%m-%d')
df['HASH_AR_ID'] = df['HASH_AR_ID'].astype(str)

target = dsp.read_table('group_payment_target', 'feature', use_schema=True)

prep = pd.merge(target[['HASH_AR_ID', 'FileDate', 'Previous_FileDate', 'Next_FileDate']], 
                df, on=['HASH_AR_ID'], how ='inner', suffixes=['','_y'])

print(prep['POSN_DT'].min())
print(prep['POSN_DT'].max())

df = prep.loc[(prep['POSN_DT'] < prep['FileDate']) & (prep['POSN_DT'] >= prep['Previous_FileDate'])]

print("Merge coverage")
print(1.0*len(df.index)/len(target.index))

dsp.write_table(df, 'sum_cc_bhvr_scor', 'feature')

print(len(target.index))
print(df['FileDate'].min())
print(df['FileDate'].max())