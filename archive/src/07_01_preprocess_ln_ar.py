from dsproject import dsproject
from preprocess import *
from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

dsp = dsproject()
min_date_text = str(min_date_train)[:10]
max_date_text = str(max_date_test)[:10]

min_date = datetime.strptime(min_date_text, '%Y-%m-%d')
max_date = datetime.strptime(max_date_text, '%Y-%m-%d')

df = pd.DataFrame()

for chunk in pd.read_csv(os.path.join(dsp.data_directory, 'EDW_SOR_LN_AR.txt'),delimiter='|', usecols=['POSN_DT'], chunksize=500000):

    chunk.columns = ['POSN_DT']
    chunk['POSN_DT'] = parse_dates(chunk['POSN_DT'], '%Y-%m-%d')
    df_train = filter_dates_1(chunk, min_date, max_date, date_column='POSN_DT')
    if not df_train.empty:
        if df.empty:
            df = df_train
        else:
            df = pd.concat([df, df_train])

print (len(df.index))
print (df['POSN_DT'].min())
print (df['POSN_DT'].max())
print(dsp.read_data('EDW_SOR_LN_AR', nrows=5))

index_to_row = [i+1 for i in df.index]
skip=list(set(range(1, list(chunk.index)[-1]))-set(list(index_to_row)))

use_col = ['HASH_IP_ID','HASH_AR_ID','DLQ_DYS','CTR_AMT','OTSND_BAL_AMT','ACR_INT_AMT','PNP_AMT','EFF_INT_RATE','AUTO_DB_F','PRVN_CLT_AMT','INT_ARS_AMT','CST_LOAN_ST_TP1_ID','CST_LOAN_ST_TP2_ID','CNTL_LMT_F','CLT_AMT','NPL_F','POSN_DT']

prep = dsp.read_data('EDW_SOR_LN_AR', skiprows=skip)
prep = prep[use_col]

print(list(prep.columns))
print(prep)
print(prep['POSN_DT'])
print(prep['POSN_DT'].min())
print(prep['POSN_DT'].max())
print(prep.shape)

prep.sort_values(by='POSN_DT').head()
prep.head()

dsp.write_table(prep, 'sor_ln_ar_filtered_' + min_date_text + '_' + max_date_text, 'preprocess')