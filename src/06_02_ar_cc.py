from dsproject import dsproject
from preprocess import *
from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label
dsp = dsproject()

min_date_text = str(min_date_train)[:10]
max_date_text = str(max_date_test)[:10]

df = dsp.read_table('sor_cc_ar_filtered_' + min_date_text + '_' + max_date_text, 'preprocess', index_col=0)
print(df.shape)

df0 = df
df = df0
df = delete_missing_columns(df0)
df = delete_single_value_columns(df)

df['HASH_AR_ID'] = df['HASH_AR_ID'].astype(str)
df['POSN_DT'] = parse_dates(df['POSN_DT'], '%Y-%m-%d')

target = dsp.read_table('group_payment_target', 'feature', use_schema=True)
print(target['FileDate'].min())
print(target['FileDate'].max())

print(df['POSN_DT'].min())
print(df['POSN_DT'].max())

prep = pd.merge(target[['HASH_AR_ID', 'FileDate', 'Previous_FileDate', 'Next_FileDate']], 
                df, on=['HASH_AR_ID'], how ='inner', suffixes=['','_y'])

df = prep.loc[(prep['POSN_DT'] < prep['FileDate']) & (prep['POSN_DT'] >= prep['Previous_FileDate'])]
print("Merge coverage")
print(1.0*len(df.index)/len(target.index))

print(df['FileDate'].min())
print(df['FileDate'].max())
print(dsp.data_directory)

df_train, df_test = filter_dates(df, min_date_train, max_date_train, min_date_test, max_date_test)

print("train - test index")
print(len(df_train.index))
print(len(df_test.index))

dsp.write_table(pd.concat([df_train, df_test]), 'sor_cc_ar_' + min_date_text + '_' + max_date_text, 'feature')



