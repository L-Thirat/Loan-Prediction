
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


# In[5]:

doc1 = '201701_201706'
doc2 = '201706'
doc_out = '201701_201707'
file_path_1 = os.path.join(os.path.dirname(dsp.data_directory), doc1, 'DB_SOC.txt')
file_path_2 = os.path.join(os.path.dirname(dsp.data_directory), doc2, 'DB_SOC.txt')


# In[6]:

df1 = pd.read_csv(file_path_1, delimiter="|")
df2 = pd.read_csv(file_path_2, delimiter="|")


# In[9]:

df = pd.concat([df1,df2])
df = df.reset_index(drop = True)
print (df.head())
print df['FileDate'].min()
print df['FileDate'].max()


# In[9]:

file_path = os.path.join(os.path.dirname(dsp.data_directory), doc_out, 'DB_SOC.txt')


# In[12]:

df.to_csv(file_path, sep='|')


# In[ ]:



