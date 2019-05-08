# -*- coding: utf-8 -*-
import pandas as pd
import re


def drop_(fea_df):
    '''
    drop掉'day\d'特征
    '''
    match_list = list(fea_df.columns)
    
    for col in match_list:
        if re.search(r'day?',col) != None:
            fea_df = fea_df.drop([col], axis=1)
    for col in match_list:
        if re.search(r'trans_amt?',col) != None:
            fea_df = fea_df.drop([col], axis=1)
    
    for col in match_list:
        if re.search(r'bal?',col) != None:
            fea_df = fea_df.drop([col], axis=1)
    
    return fea_df

def get_drop_day_trans_amt_bal():
    orig_bb_train = pd.read_csv('../Feature/bb_fea_train_clean.csv')
    orig_bb_test = pd.read_csv('../Feature/bb_fea_test_clean.csv')
    
    bb_train_drop = drop_(orig_bb_train)
    bb_test_drop = drop_(orig_bb_test)
    
    bb_train_drop.to_csv('../Feature/bb_fea_train_clean_beDrop.csv', index = False)
    bb_test_drop.to_csv('../Feature/bb_fea_test_clean_beDrop.csv', index = False)
    
    