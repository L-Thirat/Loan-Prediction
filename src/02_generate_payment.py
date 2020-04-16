from dsproject import dsproject
from preprocess import *
import os

dsp = dsproject()

f = open(os.path.join(dsp.data_directory, 'EDW_SOR_TXN_CC_Header.txt'),'r')
header = f.read().split('|')
'''
j=0
for chunk in pd.read_csv(os.path.join(dsp.data_directory, 'EDW_SOR_TXN_CC.txt'),usecols=[0,1,9,13,18], chunksize=1000000, delimiter='|', dtype={'HASH_AR_ID':object,'HASH_IP_ID':object}):
    j = j+1
    chunk.columns = ['HASH_IP_ID','HASH_AR_ID','TXN_CODE','TXN_VAL_DT','NET_CASH_FLOW_AMT']
    txn_cc = chunk.loc[chunk['TXN_CODE'].isin([19,20])]
    dsp.write_table(txn_cc, 'SOR_TXN_CC_filtered_' + str(j), 'preprocess', gen_schema=False)
'''
chunks = []
for i in range(1, 1000):
    chunk = dsp.read_table('SOR_TXN_CC_filtered_' + str(i), 'preprocess', index_col=0)
    if chunk.empty:
        break
    else:
        chunks.append(chunk)
df = pd.concat(chunks)

# find the best filedate for each payment
df['TXN_VAL_DT'] = parse_dates(df['TXN_VAL_DT'])
print("Max-Min TXN_VAL_DT")
print(df['TXN_VAL_DT'].max())
print(df['TXN_VAL_DT'].min())

unique_filedate = dsp.read_table('unique_filedate', 'feature', index_col=0, parse_dates=['FileDate', 'Next_FileDate', 'Previous_FileDate'])
df = df.merge(unique_filedate, on=['HASH_AR_ID'])
df0 = df.copy()
df = df.loc[((df['TXN_VAL_DT'] >= df['FileDate']) & (df['TXN_VAL_DT'] < df['Next_FileDate'])) | ((df['TXN_VAL_DT'] >= df['FileDate']) & (df['Next_FileDate'].isnull())) ]

print("Max-Min FileDate after merging")
print(df['FileDate'].min())
print(df['FileDate'].max())

payment = df.groupby(['HASH_AR_ID', 'FileDate'])['NET_CASH_FLOW_AMT'].sum().reset_index()
payment.columns = ['HASH_AR_ID', 'FileDate', 'Payment']

test_payment = df0[['HASH_AR_ID','FileDate','TXN_VAL_DT']].loc[df0['FileDate'] > df0['TXN_VAL_DT'].max()]
del test_payment['TXN_VAL_DT']
test_payment['Payment'] = np.NAN
print(test_payment)
payment = pd.concat([payment, test_payment])
payment = payment.reset_index(drop=True)

print(payment)
payment['HASH_AR_ID'] = payment['HASH_AR_ID'].astype(str)

dsp.write_table(payment, 'payment', 'feature')
group_target = dsp.read_table('group_target', 'feature', index_col=0, use_schema=True)
df = pd.merge(group_target, payment, on=['HASH_AR_ID', 'FileDate'], how ='left')

print("Merge coverage")
print(len(df.index)/len(group_target.index))

df['Payment'] = df['Payment'].fillna(0)
df['Percentage'] = df['Payment']/df['OSAMT']
df['Percentage'].loc[df['Percentage']>1.0] = 1.0

df = df[['HASH_AR_ID', 'HASH_IP_ID', 'HASH_LPM_CST_ID', 
         'FileDate', 'Previous_FileDate', 'Next_FileDate', 
         'Group', 'OSAMT', 'Payment', 'Percentage']]

dsp.write_table(df, 'group_payment_target', 'feature', 
                ds_type=['id', 'id', 'id', 'date', 'date', 'date', 
                         'category', 'numeric', 'numeric', 'numeric'])

print(len(df.index))
print("Payment is null ?")
print(df['Payment'].isnull().sum())
