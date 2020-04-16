from dsproject import dsproject
from preprocess import *
from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

dsp = dsproject()

target0 = dsp.read_table('group_payment_target', 'feature', use_schema=True)
features = ['dbsoc', 'cst_dim', 'sum_cc_bhvr_scor', 'sor_cc_ar_2016-03-01_2016-10-01']#, 'sor_ln_ar_2016-03-01_2016-10-01'] #'sor_cc_ar_20170201_20170501', #<< SWITCH
train_set, test_set = filter_dates(target0, min_date_train, max_date_train, min_date_test, max_date_test)#<< SWITCH

for feature in features:
    print('Merging feature ' + feature)
    df0 = dsp.read_table(feature, 'feature', index_col=0, dtype={'HASH_AR_ID': str}, parse_dates=['FileDate'])
    df0 = df0.drop_duplicates()
    print(df0.shape)
    df_train, df_test = filter_dates(df0, min_date_train, max_date_train, min_date_test, max_date_test)
    train_set = train_set.merge(df_train, on=['HASH_AR_ID', 'FileDate'], how='inner', suffixes=['', '_y'])
    test_set = test_set.merge(df_test, on=['HASH_AR_ID', 'FileDate'], how='inner', suffixes=['', '_y'])
    train_set = delete_excess_columns(train_set)
    test_set = delete_excess_columns(test_set)
    print(len(train_set.index))
    print(len(test_set.index))

train_set = train_set.drop_duplicates()
test_set = test_set.drop_duplicates()

dsp.write_table(train_set, 'train_' + train_label, 'feature')
dsp.write_table(test_set, 'test_' + test_label, 'feature')

