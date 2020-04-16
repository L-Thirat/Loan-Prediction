from dsproject import dsproject
from preprocess import *

dsp = dsproject()

df = dsp.read_data('CST_DIM', dtype={'HASH_IP_ID': str})
print("Len columns")
print(len(list(df.columns)))

df = delete_missing_columns(df)  
df = delete_single_value_columns(df)

df = df.drop(['CTC_ADR_PROV_TX', 'CTC_ADR_PROV_TX', 'SALUT_TH', 'OFFC_ADR_PROV_TX'], axis=1)
df = df.drop_duplicates(subset='HASH_IP_ID')

target = dsp.read_table('group_payment_target', 'feature', use_schema=True)
prep = target[['HASH_IP_ID', 'HASH_AR_ID', 'FileDate']].merge(df, on=['HASH_IP_ID'], how ='inner' )
print("Merge coverage")
print(1.0*len(prep.index)/len(target.index))

prep = prep.drop_duplicates(subset='HASH_AR_ID')

print(prep['FileDate'].min())
print(prep['FileDate'].max())

dsp.write_table(prep, 'cst_dim', 'feature')

