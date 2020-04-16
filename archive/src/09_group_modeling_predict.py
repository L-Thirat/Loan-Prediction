
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

train = dsp.read_table('train_' + train_label + '_processed', 'feature', index_col=0, dtype={'HASH_AR_ID': str})
validate = dsp.read_table('test_' + test_label + '_processed', 'feature', index_col=0, dtype={'HASH_AR_ID': str})
# ### Feature Selection

# In[7]:

from sklearn.feature_selection import chi2, f_classif, f_regression
import numpy as np

def test_feature(x, y, func=f_classif):
    f = func(x, y)
    f_table = pd.DataFrame(np.array([list(x.columns), f[0], f[1]]).T,
                                  columns=['column_name', 'f_value', 'p_value'])
    f_table['f_value'] = f_table['f_value'].astype(np.float64)
    f_table['p_value'] = f_table['p_value'].astype(np.float64)
    col_to_drop = f_table.loc[f_table['p_value'] > 0.001]['column_name'].tolist()
    return f_table, col_to_drop

#todo if label_column == None >>> fix feature_selection_validate
def feature_selection(df,label=None, label_column=None, col_to_drop_numeric=None, col_to_drop_cat=None, func_numeric=f_classif, func_cat=chi2):
    if label==None:
        data = np.array([np.NaN]*df.shape[0]).T
        y = pd.DataFrame({label_column:data})
    else:
        y = df[label_column] # y = df null
    df_cat = df[category_columns]
    df_numeric = df[numeric_columns]
    
    if not col_to_drop_numeric:
        result_table, col_to_drop_numeric = test_feature(df_numeric,y, func_numeric)
        dsp.write_table(result_table.sort_values(by='p_value'),'test_feature_numeric_' + train_label, 'meta')
        print(col_to_drop_numeric)
    df_numeric1 = df_numeric.drop(col_to_drop_numeric, axis=1)
    
    if not col_to_drop_cat:
        result_table, col_to_drop_cat = test_feature(df_cat, y, func_cat)
        dsp.write_table(result_table.sort_values(by='p_value'),'test_feature_category_' + train_label, 'meta')
        print(col_to_drop_cat)
    df_cat1 = df_cat.drop(col_to_drop_cat, axis=1)
    
    df_cat1_dummy = pd.get_dummies(df_cat1)
    x = pd.concat([df_numeric1, df_cat1_dummy], axis=1)
    
    return x,y, col_to_drop_numeric, col_to_drop_cat


# In[8]:
x, y, col_to_drop_numeric, col_to_drop_cat = feature_selection(train,True, 'group1')

# In[9]:

# def feature_selection_validate(df, col_to_drop_numeric=None, col_to_drop_cat=None, func_numeric=f_classif,func_cat=chi2):
#     df_cat = df[category_columns]
#     df_numeric = df[numeric_columns]
#
#     df_numeric1 = df_numeric.drop(col_to_drop_numeric, axis=1)
#     df_cat1 = df_cat.drop(col_to_drop_cat, axis=1)
#
#     df_cat1_dummy = pd.get_dummies(df_cat1)
#     x = pd.concat([df_numeric1, df_cat1_dummy], axis=1)
#
#     return x

x_validate,_,_,_ = feature_selection(validate,None,'group1', col_to_drop_numeric, col_to_drop_cat)


# # Model 1: Predict Self-Cure People

# In[10]:

from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report

def test_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    yhat = model.predict(x_test)
    result = classification_report(y_test, yhat)
    print(result)
    return model

def train_classifier(x, y):
    estim = {'LR': LR(solver='sag', C=0.1, random_state=42, n_jobs=-1), 
             'RFC': RFC(n_estimators=50, n_jobs=-1, oob_score=True, min_samples_leaf=10, random_state=42)}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mod = {}
    for key, value in estim.items():
        mod[key] = test_model(value, x_train, x_test, y_train, y_test)
    return mod, (x_train, x_test, y_train, y_test)
print 'Result: Group 1 Prediction'
mod, (x_train, x_test, y_train, y_test) = train_classifier(x, y)


# In[11]:

def validate_model(model, x_validate,new_col,index_df):
    yhat = model.predict_proba(x_validate)[:,1]
    df = pd.DataFrame({'HASH_AR_ID':index_df['HASH_AR_ID'],'FileDate':index_df['FileDate'],
                       new_col:yhat})
    return df


# In[12]:
#print validate.head()
#print validate.shape
#print len(list(set(validate['HASH_AR_ID'])))
#asd
print "OUTPUT PREDICT GROUP 1"
# print validate_model(mod['RFC'], x_validate,'Group1_Prediction',validate) # Group 1 prediction
df_out = validate_model(mod['RFC'], x_validate,'Group1_Prediction',validate)
df_out = df_out.drop_duplicates()
#print df_out.shape
#print len(list(set(df_out['HASH_AR_ID'])))
#asd
# ### Remove high-chance self-cure population
# 
# In order to get people with higher chance of being self-cure, we should adjust our classification threshold so that we can set aside people with more than 70% self-cure probability. For this method, we want to focus on getting high precision for label=1. People who we label self-cure should be more very likely to self-cure.

# In[14]:

from sklearn.metrics import accuracy_score, confusion_matrix

def adjusted_threshold_performance(mod, x_test, y_test, df, threshold=0.6):
    p = mod.predict_proba(x_test)
    y_true = np.array(y_test.tolist(), dtype=int) # !!
    y_pred = np.array(p[:,1]>threshold, dtype=int) # Group 1 adjus
    result = classification_report(y_true, y_pred)
    print('Group 1 prediction performance after adjusting threshold')
    print(result)
    
    # self-cure classification metrics
    confm = confusion_matrix(y_true, y_pred)
    total_test_pop = y_test.shape[0]
    error = confm[0,1]*100/total_test_pop
    print("Self-cure confusion matrix")
    confusion_table = pd.DataFrame(confm, columns=['Predict 0', 'Predict 1'], index=['Label 0', 'Label 1'])
    print(confusion_table)

    print("Number of people tested")
    print(total_test_pop)
    print("Number of people predicted to be class 1")
    print(confm[0,1] + confm[1,1])
    print("Percent of people removed")
    print((confm[0,1]+confm[1,1])*100/total_test_pop)
    print("Percentage of mislabeled people")
    print(error)
    
    test = y_test.to_frame()
    #test.columns = ['y_true']
    #test['y_pred'] = y_pred
    true_labels = train['Group'].iloc[test.loc[(y_true==0) & (y_pred==1)].index]
    print("Percent of action-roll people who got mistakenly eliminated in this step:")
    print(true_labels.value_counts()[5]*100/total_test_pop)

adjusted_threshold_performance(mod['RFC'], x_test, y_test, train)

threshold =0.6
# In[15]:
# adjusted_threshold_performance(mod['RFC'], x_validate, y_validate, validate)
p = mod['RFC'].predict_proba(x_validate)
y_pred = np.array(p[:,1]>threshold, dtype=int) # Group 1 adjus

df_out_new = pd.DataFrame({'HASH_AR_ID':validate['HASH_AR_ID'],'FileDate':validate['FileDate'],
                       'Group1_Adjus':y_pred}).drop_duplicates()
df_out = pd.merge(df_out,df_out_new,on=['HASH_AR_ID','FileDate'])
df_out = df_out.drop_duplicates()
# # Stacked Model Group 3

# In[16]:

def adjusted_threshold_prediction(mod, x, df, threshold=0.6):
    p = mod.predict_proba(x)
    df['Group1_Adjus'] = np.array(p[:,1]>threshold, dtype=int)
    nongroup1 = df.loc[df['Group1_Adjus']==0]
    return nongroup1, df

nongroup1, train = adjusted_threshold_prediction(mod['RFC'], x, train)
nongroup1_validate, validate = adjusted_threshold_prediction(mod['RFC'], x_validate, validate)

# In[17]:

x, y, col_to_drop_numeric, col_to_drop_cat = feature_selection(nongroup1,True, 'group3')
x_validate,_,_,_  = feature_selection(nongroup1_validate,None,'group3', col_to_drop_numeric, col_to_drop_cat)
# In[18]:
print "Result : Group3 Stack"
mod3, (x_train, x_test, y_train, y_test) = train_classifier(x, y)

# In[19]:

print "OUTPUT PREDICT GROUP 3"
# print validate_model(mod3['RFC'], x_validate,'Group3_Stack',nongroup1_validate) # Group 3 stack
df_out_new = validate_model(mod3['RFC'], x_validate,'Group3_Stack',nongroup1_validate).drop_duplicates()
df_out = pd.merge(df_out,df_out_new,on=['HASH_AR_ID','FileDate'])
df_out = df_out.drop_duplicates()

# # Stacked Model Group 5

# In[20]:

x, y, col_to_drop_numeric, col_to_drop_cat = feature_selection(nongroup1,True, 'group5')
x_validate,_,_,_ = feature_selection(nongroup1_validate,None,'group5', col_to_drop_numeric, col_to_drop_cat)

# In[21]:

mod5, (x_train, x_test, y_train, y_test) = train_classifier(x, y)


# In[22]:
print "OUTPUT PREDICT GROUP 5"
# print validate_model(mod5['RFC'], x_validate,'Group5_Stack',nongroup1_validate) # Group 5 stack
df_out_new = validate_model(mod5['RFC'], x_validate,'Group5_Stack',nongroup1_validate).drop_duplicates()
df_out = pd.merge(df_out,df_out_new,on=['HASH_AR_ID','FileDate'])
df_out = df_out.drop_duplicates()

# # Voting Classifier

# In[23]:

import numpy as np

def balanced_subsample(y, label_column, size=None):
    # randomly pick the same amount of samples with label 0
    #y = df[label_column]
    subsample = []

    if size is None:
        n_smp = y.value_counts().min()
    else:
        n_smp = int(size / len(y.value_counts().index))

    for label in y.value_counts().index:
        samples = y[y == label].index.values
        index_range = range(samples.shape[0])
        indexes = np.random.choice(index_range, size=n_smp, replace=False)
        subsample += samples[indexes].tolist()

    return subsample


# In[24]:

estimators = {}
nvoters = 10
sample = balanced_subsample(y_train, 'group5')
for i in range(nvoters):
    subsample = balanced_subsample(y, '5')
    x_sub = x_train.loc[subsample]
    y_sub = y_train.loc[subsample]
    
    model = RFC(n_estimators=50, n_jobs=-1, oob_score=True, min_samples_leaf=10)
    model.fit(x_train, y_train)
    estimators[i] = model


# In[25]:

def count_votes(x, y, threshold=0.75):
    votes = []
    for i in range(nvoters):
        y_hat = estimators[i].predict(x)
        votes.append(list(y_hat))
    votes = np.array(votes, dtype=int)
    sum_votes = votes.sum(axis=0)/nvoters
    total_votes = np.array(sum_votes>threshold, dtype=int)
    accuracy_score(total_votes, y)

    result = classification_report(total_votes, y)
    print(result)


# In[26]:

threshold = 0.25
print 'Result : group5_voting train'
count_votes(x_train, y_train, threshold)
print 'Result : group5_voting test'
count_votes(x_test, y_test, threshold)
# count_votes(x_validate, y_validate, threshold)
votes = []
for i in range(nvoters):
    y_hat = estimators[i].predict(x_validate)
    votes.append(list(y_hat))
votes = np.array(votes, dtype=int)
sum_votes = votes.sum(axis=0) / nvoters
total_votes = np.array(sum_votes > threshold, dtype=int) # Group 5 voting
df_out_new = pd.DataFrame({'HASH_AR_ID':nongroup1_validate['HASH_AR_ID'],'FileDate':nongroup1_validate['FileDate'],
                       'group5_voting':total_votes}).drop_duplicates()
df_out = pd.merge(df_out,df_out_new,on=['HASH_AR_ID','FileDate'])
df_out = df_out.drop_duplicates()
# accuracy_score(total_votes, y)

# result = classification_report(total_votes, y)
# print(result)


# # Feature Importance

# ### Model 1

# In[27]:

x_group1, _, _, _ = feature_selection(train,True, 'group1')


# In[28]:

feature_importance = pd.DataFrame(np.array([list(x_group1.columns), list(mod['RFC'].feature_importances_)]).T, 
             columns=['column_name', 'feature_importance'])
feature_importance['feature_importance'] = feature_importance['feature_importance'].astype(float)


# In[29]:

print "feature important Group1"
print feature_importance.sort_values(by='feature_importance', ascending=False).head(15)


# ### Model 2

# In[30]:

x_group5, _, _, _ = feature_selection(nongroup1,True, 'group5')


# In[31]:

lis = [list(x_group5.columns)]
col_names = []
for j, es in estimators.items():
    col_names.append('feature_importance_' + str(j))
    lis.append(es.feature_importances_)

feature_importance = pd.DataFrame(np.array(lis).T, columns=['column_name']+col_names)


# In[32]:

feature_importance['feature_importance']=feature_importance[col_names].astype(float).sum(axis=1)
feature_importance = feature_importance.drop(col_names, axis=1)


# In[33]:

feature_importance.head()


# In[34]:
print "feature important Group5"
print feature_importance.sort_values(by='feature_importance', ascending=False).head(15)


# # Percentage Prediction

# In[37]:

x, y, col_to_drop_numeric, col_to_drop_cat = feature_selection(train,True, 'Percentage',
                                                              func_numeric=f_regression, func_cat=f_regression)
x_validate,_,_,_ = feature_selection(validate,None,'Percentage', col_to_drop_numeric, col_to_drop_cat,
                                               func_numeric=f_regression, func_cat=f_regression)


# In[44]:

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn.neighbors import KNeighborsRegressor as KNR
from sklearn.linear_model import LinearRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import mean_squared_error
from math import sqrt

def test_model(model, x_train, x_test, y_train, y_test):
    model.fit(x_train, y_train)
    yhat = model.predict(x_test)
    result = sqrt(mean_squared_error(y_test, yhat))
    print("Error of prediction : %2.3f" % result)
    return model

def train_regressor(x, y):
    estim = {'LR': LR(n_jobs=-1), 
             'RFR': RFR(n_estimators=50, n_jobs=-1, oob_score=True, min_samples_leaf=10, random_state=42)}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mod = {}
    for key, value in estim.items():
        mod[key] = test_model(value, x_train, x_test, y_train, y_test)
    return mod, (x_train, x_test, y_train, y_test)

#todo fix bug here


x, y, col_to_drop_numeric, col_to_drop_cat = feature_selection(nongroup1,True, 'Percentage',
                                                              func_numeric=f_regression, func_cat=f_regression)
x_validate,_,_,_ = feature_selection(nongroup1_validate,None,'Percentage', col_to_drop_numeric, col_to_drop_cat,
                                               func_numeric=f_regression, func_cat=f_regression)

modr1, (x_train, x_test, y_train, y_test) = train_regressor(x, y)
# print validate_model(modr1['RFC'], x_validate,'group5_voting',nongroup1_validate) # Percentage Prediction
def validate_model(model, x_validate,new_col,index_df):
    yhat = model.predict(x_validate)
    df = pd.DataFrame({'HASH_AR_ID':index_df['HASH_AR_ID'],'FileDate':index_df['FileDate'],
                       new_col:yhat})
    return df
df_out_new = validate_model(modr1['RFR'], x_validate,'Percentage_Prediction',nongroup1_validate).drop_duplicates()
df_out = pd.merge(df_out,df_out_new,on=['HASH_AR_ID','FileDate'])
df_out = df_out.drop_duplicates()
print "Dateframe final output"
print df_out.head()

print "group1_adjust_threshold = 0.6"
print "count_votes_threshold = 0.25"
print "WRITING OUTPUT..."
# print df_out.dtypes
dsp.write_table(df_out,'result_predict_model','output')


# In[ ]:

#modr2, (x_train, x_test, y_train, y_test) = train_regressor(x, y)


# In[ ]:



