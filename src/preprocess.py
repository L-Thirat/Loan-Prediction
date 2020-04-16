import pandas as pd
import numpy as np
rd_dict = {'1) Current': 0,'2) 1-30': 1, '3) 31-60': 2, '4) 61-90': 3,
           '5) 91-120':4, '6) 121-150':5, '7) 151-180':6, '8) 181-365':7, '9 365 up':8}
g_dict = {'1. Selfcured': 1, '2. No Action - Roll': 2, '3. Action - Save': 3, '4. Action - Roll (Have Pay)': 4,
              '5. Action - Roll (No Pay)': 5}

def transform_range_day(df):
    # generate dictionary
    u = df['Range_day'].unique().tolist()

    # change Range_day labels to numbers
    # change Group labels to numbers
    range_day_label = df['Range_day'].unique()
    for lab in range_day_label:
        df = df.replace({'Range_day': {lab: rd_dict[lab]}})
    return df


def transform_group(df):
    group_label = df['Group'].unique()
    for lab in group_label:
        df = df.replace({'Group': {lab: g_dict[lab]}})
    return df

def replace_nan(table):
    for column in table.columns:
        ratio = float(table[column].isnull().sum()) / float(table.shape[0])
        if ratio > 0:
            print(column + ': ', str(ratio))
            table[column] = table[column].fillna("Unknown")
        return table

def parse_dates(s, format=None):
    if not format:
        dates = {date: pd.to_datetime(date) for date in s.unique()}
    else:
        dates = {date: pd.to_datetime(date, format=format) for date in s.unique()}
    return s.apply(lambda v: dates[v])

def delete_missing_columns(df, ratio=0.95):
    total = len(df.index)
    deleted_columns = []
    for column in df.columns:
        if df[column].isnull().sum()/total > ratio:
            deleted_columns.append(column)
    df = df.drop(deleted_columns, axis=1)
    print("Deleted from missings list = " + str(deleted_columns))
    print("Number of column deleted = %d" % len(deleted_columns))
    return df

def delete_single_value_columns(df):
    drop_columns = []
    for column in list(df.columns):
        if df[column].unique().shape[0]==1:
            drop_columns.append(column)
    df = df.drop(drop_columns, axis=1)
    print("Delete single value column list")
    print(drop_columns)
    print("Number of column deleted")
    print(len(drop_columns))
    return df

def delete_excess_columns(df):
    drop_columns = []
    for column in list(df.columns):
        if '_y' in column:
            drop_columns.append(column)
    df = df.drop(drop_columns, axis=1)
    print("Delete single value column list")
    print(drop_columns)
    print("Number of column deleted")
    print(len(drop_columns))
    return df

from datetime import datetime

def filter_dates(df0, min_date_train, max_date_train, min_date_test, max_date_test, date_column='FileDate'):
    df = df0.loc[df0[date_column] >= min_date_train]
    df_train = df.loc[df[date_column] < max_date_train]
    df = df0.loc[df0[date_column] >= min_date_test]
    df_test = df.loc[df[date_column] < max_date_test]

    return df_train, df_test

def filter_dates_1(df0, min_date, max_date, date_column='FileDate'):
    df = df0.loc[df0[date_column] >= min_date]
    df_train = df.loc[df[date_column] < max_date]

    return df_train
