from sklearn.feature_selection import chi2, f_classif, f_regression
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.linear_model import LogisticRegression as LR
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import pickle
from archive.src.dsproject import DSProject

dsp = DSProject()

train_dataset = 'month_201605_201607_train'
validate_dataset = 'month_201609_test'
score_dataset = 'month_201608_score'

index_columns = ['Unnamed: 0']

delete_columns = ["OWN_BR", "KBNK_EMPE_ID", "PHOTO_F", "COLL_RSN_CODE", "BOT_IDY_CL_ID", "ORIG_BR_NBR", "PRJ_ID",
                  "Month"]

date_columns = ["EFF_DT", "END_DT", "BRTH_DT", "ESTB_DT", "EFF_CST_DT", "BLC_DT", "ALT_BLC_CODE_DT", "OPN_DT", "MAT_DT",
                "LAST_PRCH_DT", "LAST_ADV_DT", "LAST_CASH_PYMT_DT", "LAST_RTL_PYMT_DT", "PYMT_DUE_DT", "LAST_MNT_DT",
                "NPL_DT"]

log1p_columns = ['OSPRINCIPLE', 'PAST_DUE_LT30_AMT', 'AV_CASH_ADV_AMT_P6M', 'DLQ_DYS', 'LAST_DLQ_MO', 'TAMT_DUE',
                 'CASH_ADV_FEE_AMT', 'AR_AGE_MO', 'II_AMT', 'LAST_PRCH_DYS', 'NBR_MO_SNC_LAST_PYMT', 'OSAMT',
                 'PAST_DUE_30_AMT', 'PAST_DUE_AMT', 'RTL_AMT', 'CR_LMT_AMT']

min_max_columns = ['NBR_DLQ_ST_1_29_P3M', 'NBR_OLMT_ST_P6M', 'NBR_DYS_NOT_PY_30DYS_LAST_3MO', 'NBR_PART_PYMT_P6M',
                   'NBR_DYS_NOT_PY_30DYS_LAST_8MO', 'NBR_DYS_NOT_PY_3DYS_LAST_1MO', 'NBR_OLMT_ST_P3M',
                   'NBR_DYS_NOT_PY_8DYS_LAST_1MO', 'NBR_DLQ_ST_30_59_P3M', 'NBR_FULL_PYMT_P6M', 'AR_GOOD_BAD_SCOR',
                   'AR_RSK_GRD', 'AR_RSK_PERF_IND', 'CST_WST_RSK_GRD_IND', 'CST_WST_RSK_IND']

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
                   'INT_BILL_NOT_PAID_AMT', 'NBR_DYS_NOT_PY_30DYS_LAST_8MO', 'RTL_AMT',
                   'NBR_DLQ_ST_1_29_P3M', 'CR_LMT_AMT']

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


with open('../pkl/column_list.pickle', 'rb') as f:
    numeric_columns, category_columns = pickle.load(f)


def test_feature(x, y, func=f_classif):
    """Check important feature by p-value

    :param x: data
    :param y: label
    :param func: feature selection model
    :return: score table, no important column name
    """
    f_score = func(x, y)
    f_table = pd.DataFrame(np.array([list(x.columns), f_score[0], f_score[1]]).T,
                           columns=['column_name', 'f_value', 'p_value'])
    f_table['f_value'] = f_table['f_value'].astype(np.float64)
    f_table['p_value'] = f_table['p_value'].astype(np.float64)
    col_to_drop = f_table.loc[f_table['p_value'] > 0.001]['column_name'].tolist()
    return f_table, col_to_drop


def feature_selection(df, label_column=None, col_to_drop_numeric=None, col_to_drop_cat=None, func_numeric=f_classif,
                      func_cat=chi2):
    """Select feature

    :param df: dataframe
    :param label_column: label name
    :param col_to_drop_numeric: drop columns in numeric data
    :param col_to_drop_cat: drop columns in category data
    :param func_numeric: numerical function
    :param func_cat: categorical function
    :return: data, label, drop columns in numeric data, drop columns in category data
    """
    if not label_column:
        y = None
    else:
        y = df[label_column]
    df_cat = df[category_columns]
    df_numeric = df[numeric_columns]

    if not col_to_drop_numeric:
        result_table, col_to_drop_numeric = test_feature(df_numeric, y, func_numeric)
        dsp.write_table(result_table.sort_values('p_value'), 'test_feature_numeric', 'meta')
        print(col_to_drop_numeric)
    df_numeric1 = df_numeric.drop(col_to_drop_numeric, axis=1)

    if not col_to_drop_cat:
        result_table, col_to_drop_cat = test_feature(df_cat, y, func_cat)
        dsp.write_table(result_table.sort_values('p_value'), 'test_feature_category', 'meta')
        print(col_to_drop_cat)
    df_cat1 = df_cat.drop(col_to_drop_cat, axis=1)

    df_cat1_dummy = pd.get_dummies(df_cat1)
    x = pd.concat([df_numeric1, df_cat1_dummy], axis=1)

    return x, y, col_to_drop_numeric, col_to_drop_cat


def test_model(model, x_train, x_test, y_train, y_test):
    """Check model

    :param model: model
    :param x_train: train data
    :param x_test: test data
    :param y_train: train label
    :param y_test: test label
    :return: model trained, prediction score
    """
    model.fit(x_train, y_train)
    yhat = model.predict(x_test)
    result = classification_report(y_test, yhat)
    print(result)
    return model, result


def train_classifier(x, y):
    """train data

    :param x: data
    :param y: label
    :return: model trained, prediction score, data set
    """
    estim = {'LR': LR(solver='sag', C=0.1, random_state=42, n_jobs=-1),
             'RFC': RFC(n_estimators=50, n_jobs=-1, oob_score=True, min_samples_leaf=10, random_state=42)}
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

    mod = {}
    result = {}
    for key, value in estim.items():
        mod[key], result[key] = test_model(value, x_train, x_test, y_train, y_test)

    return mod, result, (x_train, x_test, y_train, y_test)


def validate_model(model, x_validate, y_validate):
    """validate model

    :param model: model
    :param x_validate: validation data
    :param y_validate: validation label
    :return: prediction score
    """
    yhat = model.predict(x_validate)
    result = classification_report(y_validate, yhat)
    print(result)
    return result


def make_feature_importance_table(model, x, file_name):
    """create feature importance table

    :param model: model
    :param x: data
    :param file_name: file name
    :return: None
    """
    feature_importance = pd.DataFrame(np.array([list(x.columns), list(model.feature_importances_)]).T,
                                      columns=['column_name', 'feature_importance'])
    feature_importance['feature_importance'] = feature_importance['feature_importance'].astype(float)
    print(feature_importance.sort_values('feature_importance', ascending=False).head(15))
    dsp.write_table(feature_importance, file_name, 'meta')
