from dsproject import dsproject
from preprocess import *
from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

dsp = dsproject()

min_date_text = str(min_date_train)[:10]
max_date_text = str(max_date_test)[:10]

df = dsp.read_table('sor_ln_ar_filtered_' + min_date_text + '_' + max_date_text, 'preprocess', index_col=0)
print(df.shape)
df0 = df

df = delete_missing_columns(df0)
df = delete_single_value_columns(df)
df.head()
target = dsp.read_table('group_payment_target', 'feature', use_schema=True)

df['HASH_AR_ID'] = df['HASH_AR_ID'].astype(str)
df['POSN_DT'] = parse_dates(df['POSN_DT'], '%Y-%m-%d')
prep = pd.merge(target[['HASH_AR_ID', 'FileDate', 'Previous_FileDate', 'Next_FileDate']], 
                df, on=['HASH_AR_ID'], how ='inner', suffixes=['','_y'])

df = prep.loc[(prep['POSN_DT'] < prep['FileDate']) & (prep['POSN_DT'] >= prep['Previous_FileDate'])]
print(df.head())
print("Merge coverage")
print(1.0*len(df.index)/len(target.index))

df_new = pd.DataFrame({'NBR_LN_AR':df.groupby( ['HASH_IP_ID','FileDate'] )['HASH_AR_ID'].nunique(),
                     'TOTAL_LN_AMT':df.groupby( ['HASH_IP_ID','FileDate'] )['OTSND_BAL_AMT'].sum(),
                   'TOTAL_DLQ_DYS':df.groupby( ['HASH_IP_ID','FileDate'] )['DLQ_DYS'].sum()}
                      ).reset_index()

idx = df.groupby(['HASH_IP_ID','FileDate'])['OTSND_BAL_AMT'].transform(max) == df['OTSND_BAL_AMT']
df = pd.merge(df[idx],df_new,on=['HASH_IP_ID','FileDate'],how='inner')
df_train, df_test = filter_dates(df, min_date_train, max_date_train, min_date_test, max_date_test)
print(len(df_train.index))
print(len(df_test.index))
dsp.write_table(pd.concat([df_train, df_test]), 'sor_ln_ar_' + min_date_text + '_' + max_date_text, 'feature')

