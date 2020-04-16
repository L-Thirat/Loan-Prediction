from dsproject import dsproject
from preprocess import *
from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

dsp = dsproject()

min_date = min_date_train
max_date = max_date_test

df0 = dsp.read_data('EDW_SOR_CC_AR', nrows=10)

df = pd.DataFrame()
j=0
for chunk in dsp.read_data('EDW_SOR_CC_AR', usecols=[3], chunksize=1000000):
    j=j+1
    chunk.columns = ['POSN_DT']
    chunk['POSN_DT'] = parse_dates(chunk['POSN_DT'], '%Y-%m-%d')
    df_train = filter_dates_1(chunk, min_date, max_date, date_column='POSN_DT')
    if not df_train.empty:
        if df.empty:
            df = df_train
        else:
            df = pd.concat([df, df_train])
        print('Processing chunk ' + str(j) + '. Data found.')
    else:
        print('Processing chunk ' + str(j) + '. Data not found.')

print("Min-Max Date : POSN_DT")
print(df['POSN_DT'].min())
print(df['POSN_DT'].max())
index_to_row = [i+1 for i in df.index]
skip=list(set(range(1, list(chunk.index)[-1]))-set(list(index_to_row)))

prep = dsp.read_data('EDW_SOR_CC_AR', skiprows=skip)

print(prep)
print(prep['POSN_DT'].min())
print(prep['POSN_DT'].max())
print("Number of rows in EDW_SOR_CC_AR")
print(len(prep.index))

dsp.write_table(prep, 'sor_cc_ar_filtered_' + min_date_text + '_' + max_date_text, 'preprocess')