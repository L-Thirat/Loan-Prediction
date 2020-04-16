
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
EDW = ['DB_SOC.txt','EDW_LPM_X_AR_X_CIS.txt','EDW_SOR_TXN_CC.txt','CST_DIM.txt',
        'EDW_SUM_CC_AR_BHVR_SCOR.txt','EDW_SOR_CC_AR.txt','EDW_SOR_LN_AR.txt']

# EDW = ['EDW_SOR_LN_AR.txt']

fol1 = os.path.join(os.path.dirname(dsp.class_directory), 'data','201701_201706')
fol2 = os.path.join(os.path.dirname(dsp.class_directory), 'data','201706')
folder = [fol1,fol2]

for files in EDW:
    print files
    if files not in ['EDW_SOR_TXN_CC.txt', 'EDW_SUM_CC_AR_BHVR_SCOR_N.txt', 'EDW_SOR_CC_AR.txt','CST_DIM.txt','EDW_SOR_LN_AR.txt']:
        df_fol1 = pd.read_csv((fol1+'/%s'%(files)), delimiter="|", encoding = "ISO-8859-1")
        df_fol2 = pd.read_csv((fol2+'/%s'%(files)), delimiter="|", encoding = "ISO-8859-1")
    else:
        df_fol1 = pd.read_csv((fol1+'/%s'%(files)), delimiter="|")
        df_fol2 = pd.read_csv((fol2+'/%s'%(files)), delimiter="|")
    try:
        print 'FileDate'
        print df_fol1['FileDate'].min()
        print df_fol2['FileDate'].min()
        print df_fol1['FileDate'].max()
        print df_fol2['FileDate'].max()
    except:pass
    try:
        print 'POSN_DT'
        print df_fol1['POSN_DT'].min()
        print df_fol2['POSN_DT'].min()
        print df_fol1['POSN_DT'].max()
        print df_fol2['POSN_DT'].max()
    except:pass
    print "#############################"
