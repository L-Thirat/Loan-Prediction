
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

EDW = ['EDW_LPM_X_AR_X_CIS.txt','EDW_SOR_TXN_CC.txt','CST_DIM.txt',
       'EDW_SUM_CC_AR_BHVR_SCOR.txt','EDW_SOR_CC_AR.txt','EDW_SOR_LN_AR.txt']

# EDW = ['EDW_SOR_AR_CC.txt']

# In[3]:

path_data = os.path.join(os.path.dirname(dsp.class_directory), 'data')
print path_data


# In[ ]:

fol1 = '201701_201706'
fol2 = '201706'
for files in EDW:
    print files
    if files not in ['EDW_SOR_TXN_CC.txt', 'EDW_SUM_CC_AR_BHVR_SCOR_N.txt', 'EDW_SOR_CC_AR.txt','CST_DIM.txt','EDW_SOR_LN_AR.txt']:
        df_fol1 = pd.read_csv((path_data+'/%s/%s'%(fol1,files)), delimiter="|", encoding = "ISO-8859-1")
        df_fol2 = pd.read_csv((path_data+'/%s/%s'%(fol2,files)), delimiter="|", encoding = "ISO-8859-1")
    else:
        df_fol1 = pd.read_csv((path_data+'/%s/%s'%(fol1,files)), delimiter="|")
        df_fol2 = pd.read_csv((path_data+'/%s/%s'%(fol2,files)), delimiter="|")
    df_out = pd.concat([df_fol1,df_fol2])
    df_out = df_out.reset_index(drop = True)

    file_path = os.path.join(dsp.data_directory, files)
    df_out.to_csv(file_path, sep='|')


# In[ ]:



