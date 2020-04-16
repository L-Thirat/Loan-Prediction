
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

import pickle
with open('../pkl/column_list.pickle', 'rb') as f:
    numeric_columns, category_columns = pickle.load(f)


# In[3]:

from dsproject import min_date_train, max_date_train, min_date_test, max_date_test, train_label, test_label

df = dsp.read_table('result_predict_model', 'output', index_col=0, dtype={'HASH_AR_ID': str})

df_key = ['Group1_Prediction','Group3_Stack','Group5_Stack','Percentage_Prediction']
for keyword in df_key:
    df[keyword][df[keyword]>=0.5] = 1
    df[keyword][df[keyword]<0.5] = 0

df = df.drop_duplicates()

print df.head()
print df[df['Group1_Adjus']==1].count()
print df[df['group5_voting']==1].count()
print df.shape
print len(list(set(df['HASH_AR_ID'])))
print len(list(set(df['FileDate'])))

'''
(171156, 8)
4252
1
'''