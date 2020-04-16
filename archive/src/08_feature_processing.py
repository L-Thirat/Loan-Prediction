
# coding: utf-8

# In[1]:

# This chunk of code make it possible to use src functions
import sys
import os
module_path = os.path.abspath(os.path.join('../src'))

if module_path not in sys.path:
    sys.path.append(module_path)
import warnings
warnings.filterwarnings('ignore')

from dsproject import dsproject
from preprocess import *

dsp = dsproject()


# In[2]:

numeric_columns = ['NBR_DLQ_ST_30_59_P3M', 'OSPRINCIPLE', 'AR_RSK_SCOR', 'AR_RSK_GRD', 'AV_CASH_ADV_AMT_P6M',
                       'DLQ_DYS', 'OTSND_BAL_AMT', 'PAST_DUE_LT30_AMT', 'BAL_LMT_PCT', 'CST_WST_SCOR', 'LAST_DLQ_MO',
                       'NBR_DYS_NOT_PY_3DYS_LAST_1MO', 'CST_BAL_AV_LMT_3MO_PCT', 'TAMT_DUE', 'CASH_ADV_AV_LMT_3MO_PCT',
                       'PYMT_PREV_BAL_AV_3MO_PCT', 'CASH_ADV_FEE_AMT', 'CRN_CYC_AMT', 'AR_GOOD_BAD_SCOR',
                       'NBR_DYS_NOT_PY_30DYS_LAST_3MO', 'ACR_INT_CASH_ADV_YTD_AMT', 'NBR_PART_PYMT_P6M',
                       'BAL_AV_LMT_3MO_PCT', 'AR_AGE_MO', 'ACR_INT_AMT', 'II_AMT', 'CST_WST_RSK_IND', 'LAST_PRCH_DYS',
                       'AV_LMT_USG_RTO_P6M', 'ACR_INT_CASH_ADV_AMT', 'NBR_MO_SNC_LAST_PYMT', 'NBR_OLMT_ST_P6M',
                       'ACR_INT_RTL_YTD_AMT', 'OSAMT', 'CRN_STMT_BAL', 'PAST_DUE_30_AMT', 'PAST_DUE_AMT',
                       'NBR_FULL_PYMT_P6M', 'NBR_OLMT_ST_P3M', 'NBR_DYS_NOT_PY_8DYS_LAST_1MO', 'PREV_STMT_BAL',
                       'AR_RSK_PERF_IND', 'CASH_ADV_AMT', 'ACR_INT_RTL_AMT', 'NON_AUTO_PYMT_AMT', 'AV_PYMT_RTO_P6M',
                       'OWN_BR', 'INT_BILL_NOT_PAID_AMT', 'NBR_DYS_NOT_PY_30DYS_LAST_8MO', 'RTL_AMT',
                       'NBR_DLQ_ST_1_29_P3M', 'CR_LMT_AMT']


# In[3]:

min_max_columns = ['NBR_DLQ_ST_1_29_P3M', 'NBR_OLMT_ST_P6M', 'NBR_DYS_NOT_PY_30DYS_LAST_3MO', 'NBR_PART_PYMT_P6M',
                       'NBR_DYS_NOT_PY_30DYS_LAST_8MO', 'NBR_DYS_NOT_PY_3DYS_LAST_1MO', 'NBR_OLMT_ST_P3M',
                       'NBR_DYS_NOT_PY_8DYS_LAST_1MO', 'NBR_DLQ_ST_30_59_P3M', 'NBR_FULL_PYMT_P6M', 'AR_GOOD_BAD_SCOR',
                       'AR_RSK_GRD', 'AR_RSK_PERF_IND', 'CST_WST_RSK_GRD_IND', 'CST_WST_RSK_IND']


# In[4]:

log1p_columns = ['OSPRINCIPLE', 'PAST_DUE_LT30_AMT', 'AV_CASH_ADV_AMT_P6M', 'DLQ_DYS', 'LAST_DLQ_MO', 'TAMT_DUE',
                     'CASH_ADV_FEE_AMT', 'AR_AGE_MO', 'II_AMT', 'LAST_PRCH_DYS', 'NBR_MO_SNC_LAST_PYMT', 'OSAMT',
                     'PAST_DUE_30_AMT', 'PAST_DUE_AMT', 'RTL_AMT', 'CR_LMT_AMT']


# In[5]:

category_columns = ['Card Type New', 'PortFolio', 'BILLCycle', 'ProductName', 'Range_day',
                        'BOT_IDY_CL_ID', 'ORIG_BR_NBR', 'KBNK_EMPE_ID', 'CTC_ADR_RGON_ID',
                        'CC_STMT_RET_RSN_ID', 'SALUT_EN', 'CR_ASES_ID', 'E_MAIL_F',
                        'OFFC_ADR_LO_ID', 'CST_TP_ID', 'OCP_ID', 'IDV_PROF_TP_ID',
                        'ED_LVL_ID', 'MAR_ST_TP_ID', 'KBNK_STFF_F', 'CST_DUAL_SEG_ID',
                        'CTC_ADR_LO_ID', 'IDENTN_TP_ID', 'CST_DUAL_SUB_SEG_ID', 'GND_ID', 'CONSND_KL_F',
                        'OFFC_ADR_RGON_ID', 'RACE_ID', 'MBL_PH_F', 'KBNK_IDY_CL_ID',
                        'OCP_GRP_ID', 'CST_SEG_ID', 'IP_LCS_TP_ID', 'INCM_RNG_ID', 'CST_SUB_SEG_ID',
                        'PRVT_WLTH_F', 'NAT_ID', 'BILL_CYC_ID', 'CARD_LVL', 'AR_RSK_SCOR_ID',
                        'AR_GOOD_BAD_CODE', 'CC_TP_ID', 'CRN_DLQ_ST', 'IS_INACT_ST_P8M_F',
                        'CST_WRST_DLQ_ST', 'CARD_TP', 'PRJ_ID', 'AFF_MBR_ORG_ID', 'DLQ_DAY_ITRV_ID',
                        'AFF_CODE', 'DLQ_ST_ID2', 'PAST_DUE_AMT_RNG_ID', 'DLQ_ST_ID7', 'FSVC_PDA_SEG_ID',
                        'DLQ_ST_ID15', 'IS_PNP_CARD_F', 'NBR_RQS', 'DLQ_ST_ID4', 'DLQ_ST_ID18', 'DLQ_ST_ID5',
                        'PHOTO_F', 'COLL_RSN_CODE', 'IS_GOOD_PYMT_F', 'DLQ_ST_ID6', 'COLL_BR_NBR',
                        'CR_LMT_AMT_RNG_ID', 'LAST_PYMT_ITRV_ID', 'NPL_F', 'DLQ_ST_ID24', 'MISC_CODE',
                        'DLQ_ST_ID17', 'IS_CARD_VLD_F', 'CC_ST_ID', 'CST_TP_ID', 'DLQ_ST_ID19', 'MN_PYMT_F',
                        'NBR_PNP_CARD', 'CARD_TP', 'AMT_RNG_ID', 'DLQ_ST_ID20', 'ACT_PYMT_ITRV_ID', 'DLQ_ST_ID3',
                        'IS_CLCB_F', 'AR_SEG_SZ', 'COLL_ID', 'AU_ID', 'ALT_BLC_CODE_ID', 'DLQ_ST_ID11',
                        'DLQ_ST_ID16', 'DLQ_ST_ID10', 'DLQ_ST_ID23', 'DLQ_ST_ID14', 'ALT_CST_ORG_NBR',
                        'CRN_BAL_AMT_RNG_ID', 'PNP_CARD_ITRV_ID', 'BILL_CYC_ID', 'PYMT_MTH_ID', 'DLQ_ST_ID',
                        'DLQ_ST_ID21', 'DLQ_ST_ID9', 'DLQ_ST_ID1', 'AC_GRP_ID', 'CARD_TP_ID', 'IS_STFF_F',
                        'DLQ_ST_ID12', 'CC_TP_ID', 'DLQ_ST_ID13', 'RSPL_DEPT_ID', 'DLQ_ST_ID8', 'DLQ_ST_ID22',
                        'BLC_CODE_ID', 'AFF_MBR_TP_ID', 'CARD_CGY_ID']


# In[6]:

min_max_columns = list(set(min_max_columns))
log1p_columns = list(set(log1p_columns))
category_columns = list(set(category_columns))


# In[7]:

from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

df_train = dsp.read_table('train_' + train_label, 'feature', index_col=0)


# In[8]:
df_train_col = list(df_train.columns)

def check_del_col(list_type_col):
    pass_col = []
    for item in list_type_col:
        if item in df_train_col:
            pass_col.append(item)
    return pass_col

numeric_columns = check_del_col(numeric_columns)
min_max_columns = check_del_col(min_max_columns)
log1p_columns = check_del_col(log1p_columns)
category_columns = check_del_col(category_columns)

df_test = dsp.read_table('test_' + test_label, 'feature', index_col=0)


# In[9]:

col_to_drop = []
for column in category_columns:
    n_unique = len(df_train[column].unique().tolist())
    if n_unique>200:
        print((column, len(df_train[column].unique().tolist())))
        col_to_drop.append(column)
df_train = df_train.drop(col_to_drop, axis=1)
df_test = df_test.drop(col_to_drop, axis=1)
category_columns = list(set(category_columns) - set(col_to_drop))


# In[10]:

total = len(df_train.index)
col_to_drop = []
for column in df_train.columns:
    if df_train[column].isnull().sum()/total > 0.95:
        col_to_drop.append(column)
df_train = df_train.drop(col_to_drop, axis=1)
df_test = df_test.drop(col_to_drop, axis=1)


# In[11]:

from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np

def feature_processing(df):
    # numeric columns
    for column in log1p_columns:
        df[column] = np.log(df[column]+1)
    for column in min_max_columns:
        df[column] = df[column]/df[column].unique().max()
    for column in numeric_columns:
        df[column] = df[column].fillna(df[column].dropna().mean())
    
    scaler = StandardScaler().fit(df[numeric_columns])
    df[numeric_columns] = scaler.transform(df[numeric_columns])
    
    # categorical columns
    labenc = preprocessing.LabelEncoder()
    for column in category_columns:
        df[column] = labenc.fit_transform(df[column].tolist())
    return df


# In[12]:

print df_train.head()

# In[13]:

df_train = feature_processing(df_train)


# In[14]:

df_test = feature_processing(df_test)


# In[15]:

target = dsp.read_table('group_payment_target', 'feature', use_schema=True)


# In[16]:

target['FileDate'] = target['FileDate'].astype(str)
df_train['HASH_AR_ID'] = df_train['HASH_AR_ID'].astype(str)
df_test['HASH_AR_ID'] = df_test['HASH_AR_ID'].astype(str)


# In[17]:

def create_dataset(df):
    
    merged = df.merge(target, on=['HASH_AR_ID', 'FileDate'], suffixes=['', '_y'])
    merged[['group1', 'group3', 'group5']] = pd.get_dummies(merged['Group'])
    return merged

train = create_dataset(df_train)
# validate = create_dataset(df_test)


# In[18]:

dsp.write_table(train, 'train_' + train_label + '_processed', 'feature')
# dsp.write_table(validate, 'test_' + test_label + '_processed', 'feature')
dsp.write_table(df_test, 'test_' + test_label + '_processed', 'feature')


# In[19]:

import pickle

with open('../pkl/column_list.pickle', 'wb') as f:
    pickle.dump([numeric_columns, category_columns], f)


# In[20]:

len(train.index)


# In[21]:

# len(validate.index)


# In[ ]:



