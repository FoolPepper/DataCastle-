# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:46:52 2018

@author: FoolPepper

"""
import pandas as pd
import numpy as np
import time

def get_cleaned_data():
    """return 训练集两组，测试集两组， label和预测值dataFrame"""
    """获得原始数据"""
    print('Cleaned data Reading...(final)')
    op_train = pd.read_csv('../CleanedData/op_train_deal.csv')
    trans_train = pd.read_csv('../CleanedData/trans_train_deal.csv')
    
    op_test = pd.read_csv('../CleanedData/op_test_deal.csv')
    trans_test = pd.read_csv('../CleanedData/trans_test_deal.csv')
    y = pd.read_csv('../CleanedData/tag_train_new.csv')
    sub = pd.read_csv('../CleanedData/submit_example.csv')
    print('Cleaned Data Reading is done!')
    return op_train, trans_train, op_test, trans_test, y, sub

def time_window(trans, op, label):
    """ 时间内是否有变化与变动,相同时间 """
    """ 输入未处理过的 trans和op """   
    for deliver in ['device2', 'ip1', 'mac1','mac2','wifi','device_code1']:     #选取
        op[deliver + '_UID_nums_per_second'] = op.groupby(['UID','time'])[deliver].transform('nunique')
        
        temp = op.groupby('UID')[deliver + '_UID_nums_per_second'].max().reset_index(name = deliver + '_max_UID_nums_per_second_op')
        label = pd.merge(label, temp, on = 'UID', how = 'left').fillna(-1)
    
    for deliver in ['merchant', 'acc_id1', 'device1', 'device2', 'device_code1', 'device_code3', 'mac1', 'ip1']:
        trans[deliver + '_UID_nums_per_second'] = trans.groupby(['UID','time'])[deliver].transform('nunique')

        temp = trans.groupby('UID')[deliver + '_UID_nums_per_second'].max().reset_index(name = deliver + '_UID_nums_per_second')
        label = pd.merge(label, temp, on = 'UID', how = 'left').fillna(-1)
    
    #feature
    return label
    
    
def UID_Used_by_different_stuff(trans, op, label):   
    """账号被不同设备使用，选取独特stuff: ip, device, wifi, mac..... """
    
    for deliver in ['mac1','ip1','mac2','device2','device_code1','wifi']:          
        op[deliver + '_diff_UID'] = op.groupby([deliver])['UID'].transform('nunique')
        temp = op.groupby(['UID'])[deliver + '_diff_UID'].median().reset_index(name = deliver + '_diff_UID_median_op')
        temp2 = op.groupby(['UID'])[deliver + '_diff_UID'].max().reset_index(name = deliver + '_diff_UID_max_op')
        
        label = pd.merge(label, temp, on = 'UID', how = 'left').fillna(-1)
        label = pd.merge(label, temp2, on = 'UID', how = 'left').fillna(-1)
        
    for deliver in ['device2','device_code1', 'device_code3','mac1','ip1']:       
        trans[deliver + '_diff_UID'] = trans.groupby([deliver])['UID'].transform('nunique')
        temp = trans.groupby(['UID'])[deliver + '_diff_UID'].median().reset_index(name = deliver + '_diff_UID_median_trans')
        temp2 = trans.groupby(['UID'])[deliver + '_diff_UID'].max().reset_index(name = deliver + '_diff_UID_max_trans')        #异常
        
        label = pd.merge(label, temp, on = 'UID', how = 'left').fillna(-1)
        label = pd.merge(label, temp2, on = 'UID', how = 'left').fillna(-1)
        
    return label

def get_feature_4():
    """  """
    op_train, trans_train, op_test, trans_test, label, sub = get_cleaned_data()          #取出洗过的数据
    
    label = label.drop(columns = 'Tag', axis = 1)
    sub = sub.drop(columns = 'Tag', axis = 1)
    #feature_1
    fea_train_1 = time_window(trans_train, op_train, label)   
    fea_test_1 = time_window(trans_test, op_test, sub) 
    
    #feature_2
    fea_train_2 = UID_Used_by_different_stuff(trans_train, op_train, label)   
    fea_test_2 = UID_Used_by_different_stuff(trans_test, op_test, sub)    
    
    fea_train_all = pd.merge(label, fea_train_1, on = 'UID', how = 'left')
    fea_train_all = pd.merge(fea_train_all, fea_train_2, on = 'UID', how = 'left')
    
    fea_test_all = pd.merge(sub, fea_test_1, on = 'UID', how = 'left')
    fea_test_all = pd.merge(fea_test_all, fea_test_2, on = 'UID', how = 'left')

    fea_train_all.to_csv('../Feature/same_time_and_muti_UID_train.csv', index = False)
    fea_test_all.to_csv('../Feature/same_time_and_muti_UID_test.csv', index = False)


if __name__ == "__main__":  
    get_feature_4()



























