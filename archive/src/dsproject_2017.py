import pandas as pd
import os
import numpy as np

table_filename = {#'DB_SOC': 'DB_SOC_201510-201610.txt',
             'DB_SOC' : 'DB_SOC.txt',
             # 'ListAction': 'ListAction 201510-201610.txt',
             # 'Promise': 'Promise 201510-201610.txt',
             # 'RiskGrade': 'RiskGrade 201510_201610.txt',
             # 'XmtStat': 'XmtStat 201510-201606.txt',
             # 'XmtStat_CISCO': 'XmtStat_CISCO 201607-201610.txt',
             'EDW_LPM_X_AR_X_CIS': 'EDW_LPM_X_AR_X_CIS.txt',
             'EDW_SOR_TXN_CC': 'EDW_SOR_TXN_CC.txt',
             'CST_DIM': 'CST_DIM.txt',
             'EDW_SUM_CC_AR_BHVR_SCOR': 'EDW_SUM_CC_AR_BHVR_SCOR.txt',
             'EDW_SOR_CC_AR': 'EDW_SOR_CC_AR.txt',
             'EDW_SOR_LN_AR': 'EDW_SOR_LN_AR.txt'}

# training configuration
from datetime import datetime
'''
### train and test dates for 1 year data ###
min_date_train = '2016-08-01'
max_date_train = '2016-09-01'
min_date_test = '2016-09-01'
max_date_test = '2016-10-01'
'''

### train and test dates for 2017 data ###
# min_date_train = '2017-01-01'
# max_date_train = '2017-06-01'
# min_date_test = '2017-06-01'
# max_date_test = '2017-07-01'

min_date_train = '2017-01-01'
max_date_train = '2017-05-03' # MAX DB_SOC[FileDate]
min_date_test = '2017-05-03' # MAX DB_SOC[FileDate]
max_date_test = '2017-07-01'
except_pp = '201706' #fol test FileDate mm


min_date_train = datetime.strptime(min_date_train, '%Y-%m-%d')
max_date_train = datetime.strptime(max_date_train, '%Y-%m-%d')
min_date_test = datetime.strptime(min_date_test, '%Y-%m-%d')
max_date_test = datetime.strptime(max_date_test, '%Y-%m-%d')

train_label = datetime.strftime(min_date_train, '%Y%m') + '_' + datetime.strftime(max_date_train, '%Y%m')
test_label = datetime.strftime(min_date_test, '%Y%m') + '_' + datetime.strftime(max_date_test, '%Y%m')


### Default directories ###
default_class_directory = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
default_data_directory = os.path.join(os.path.dirname(default_class_directory), 'data')
default_disk_directory = os.path.join(os.path.dirname(default_class_directory), 'disk')

### Current directories ###
class_directory = default_class_directory
### train and test dates for 1 year data ###
# data_directory = os.path.join(default_data_directory, '201510_201610')
# disk_directory = os.path.join(default_disk_directory, '201510_201610')

### train and test dates for 2017 data ###
data_directory = os.path.join(default_data_directory, '201701_201707')
disk_directory = os.path.join(default_disk_directory, '201701_201707')

class dsproject():
    class_directory = ''
    data_directory = ''
    disk_directory = ''
    directories = {}

    def __init__(self):
        self.except_pp_month = except_pp
        self.class_directory = class_directory
        self.data_directory = data_directory
        self.disk_directory = disk_directory
        self.directories['feature'] = os.path.join(self.disk_directory, 'feature')
        self.directories['target'] = os.path.join(self.disk_directory, 'target')
        self.directories['preprocess'] = os.path.join(self.disk_directory, 'preprocess')
        self.directories['output'] = os.path.join(self.disk_directory, 'output')
        self.directories['meta'] = os.path.join(self.disk_directory, 'meta')

    def gen_directories(self):
        dirnames = ['preprocess', 'feature', 'target', 'output', 'meta']
        for d in dirnames:
            self.gen_directory(d)

    def gen_directory(self, dirname):
        if not os.path.isdir(os.path.join(self.disk_directory, dirname)):
            os.makedirs(os.path.join(self.disk_directory, dirname))

    def read_data(self, table_name, **kwargs):
        if table_name=='XmtStat_CISCO' or table_name=='ListAction': ##Thai string** not finished
            df = pd.read_csv(os.path.join(self.data_directory, table_filename[table_name]), delimiter="|", encoding="ISO-8859-1")
        elif table_name in ['EDW_LPM_X_AR_X_CIS', 'CST_DIM']:
            df = pd.read_csv(os.path.join(self.data_directory, table_filename[table_name]), delimiter="|", **kwargs)
        elif table_name in ['EDW_SOR_TXN_CC', 'EDW_SUM_CC_AR_BHVR_SCOR_N', 'EDW_SOR_CC_AR']:
            header_name = table_name + '_Header.txt'
            header_path = os.path.join(self.data_directory, header_name)
            if os.path.isfile(header_path):
                header = pd.read_csv(header_path, delimiter="|").T
                header = list(header.index)
                df = pd.read_csv(os.path.join(self.data_directory, table_filename[table_name]), delimiter="|", header=None, **kwargs)
                df.columns = header
            else:
                df = pd.read_csv(os.path.join(self.data_directory, table_filename[table_name]), delimiter="|", **kwargs)
        else:
            df = pd.read_csv(os.path.join(self.data_directory, table_filename[table_name]), delimiter="|")
        return df

    def read_table(self, file_name, keyword, use_schema=False, **kwargs):
        file_path = os.path.join(self.directories[keyword], file_name + '.csv')
        if use_schema:
            args = self.gen_load_args(file_name, keyword)
            df = pd.read_csv(file_path, **args)
        elif os.path.isfile(file_path):
            df = pd.read_csv(file_path, **kwargs)
        else:
            df = pd.DataFrame()
        return df

    def write_table(self, table, filename, keyword, gen_schema=True, ds_type=None, **kwargs):
        file_path = os.path.join(self.directories[keyword], filename + '.csv')
        table.to_csv(file_path, encoding='utf-8', **kwargs)
        if gen_schema:
            unique_values = []
            for column in table.columns:
                u = table[column].unique().tolist()
                if len(u) > 5:
                    u = u[0:5]
                unique_values.append(u)
            if not ds_type:
                ds_type = [0] * len(table.columns)

            schema = pd.DataFrame(np.array([table.columns, table.dtypes, unique_values, ds_type]).T, columns=['column_name', 'data_type', 'unique_values', 'ds_type'])
            file_path = os.path.join(self.directories[keyword], filename + '_schema.csv')
            schema.to_csv(file_path, encoding='utf-8', **kwargs)

    def gen_load_args(self, table_name, keyword):
        s = self.read_table(table_name + '_schema', keyword)
        float_columns = s.loc[s['ds_type'].isin(['numeric'])]['column_name'].tolist()
        string_columns = s.loc[s['ds_type'].isin(['id', 'category'])]['column_name'].tolist()
        date_columns = s.loc[s['ds_type'].isin(['date'])]['column_name'].tolist()
        dtype = dict([(c, np.float64) for c in float_columns] + [(c, str) for c in string_columns])
        return {'dtype': dtype, 'index_col': 0, 'parse_dates': date_columns}

