# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 20:49:19 2018

@author: FoolPepper

#preprocess.py
"""

import pandas as pd
import numpy as np

#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
def getRawData():
    """return 训练集两组，测试集两组， label和预测值dataFrame"""
    """获得原始数据"""
    print('data Reading...(final)')
    op_train = pd.read_csv('../RawData/operation_train_new.csv')
    trans_train = pd.read_csv('../RawData/transaction_train_new.csv')
    
    op_test = pd.read_csv('../RawData/test_operation_round2.csv')
    trans_test = pd.read_csv('../RawData/test_transaction_round2.csv')
    y = pd.read_csv('../RawData/tag_train_new.csv')
    sub = pd.read_csv('../RawData/submit_example.csv')
    print('Data Reading is done!')
    return op_train, trans_train, op_test, trans_test, y, sub

def get_item_drop_list(op_train, trans_train, op_test, trans_test, op_drop_item, trans_drop_item):
    """清洗数据"""
    
    dict_all = {}
    for item in op_drop_item:
        item_in_train = list(op_train[item].unique())
        item_in_test = list(op_test[item].unique())        
        diff = [i for i in item_in_train if i not in item_in_test]
        
        dict_all[item] = diff             #压入dict
        
    for item in trans_drop_item:
        item_in_train = list(trans_train[item].unique())
        item_in_test = list(trans_test[item].unique())  
        diff = [i for i in item_in_train if i not in item_in_test]         #发现op中有nan：trans_type2
        
        #op中trans_type中有np.nan(2018/12/10)
        if item == 'trans_type2':
            diff = [i for i in diff if ~np.isnan(i)]
        dict_all[item] = diff             #压入dict
    
    return dict_all           #返回需要drop掉的行（分布一致）            
        
    
def get_A_list_deal(op, trans, _dict, op_drop_item, trans_drop_item):
    
    """ 解决 A榜和train分布不一致的问题，输入train集trans and op """
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> start >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('A榜处理代码前: op原数量： %d, trans原数量： %d'%(len(op), len(trans)))

    for deliver in op_drop_item:
        op = op.loc[~op[deliver].isin(_dict[deliver]),:]
    for deliver in trans_drop_item:
        trans = trans.loc[~trans[deliver].isin(_dict[deliver]),:]
    
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>> end >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>') 
    print('A榜处理代码后: op数量： %d, trans数量： %d'%(len(op), len(trans)))
    return op, trans

def clean_and_drop(op_train, trans_train, op_test, trans_test):
    """清洗异常值与重复值"""
    op_train = op_train.drop_duplicates()
    trans_train = trans_train.drop_duplicates()        #drop重复值
    
    op_train = op_train[(op_train['UID'] != 17520)&(op_train['UID'] != 21463)]             #明显异常值（明显的羊毛党）
    trans_train = trans_train[(trans_train['UID'] != 17520)&(trans_train['UID'] != 21463)] 
    
    op_train = op_train.drop(['ip2','ip2_sub'], axis=1)
    trans_train = trans_train.drop(['code1','code2'], axis=1)
    
    #清洗test
    op_test = op_test.drop_duplicates()
    trans_test = trans_test.drop_duplicates()         #drop重复值
    
    op_test = op_test.drop(['ip2','ip2_sub'], axis=1)
    trans_test = trans_test.drop(['code1','code2'], axis=1)
    
    return op_train, trans_train, op_test, trans_test 


#数据读取
def data_Cleaning_run():
    op_train, trans_train, op_test, trans_test, label, sub = getRawData() 
    #数据清洗
    op_train, trans_train, op_test, trans_test = clean_and_drop(op_train, trans_train, op_test, trans_test)
    
    op_drop_item = ['mode', 'os', 'version']
    trans_drop_item = ['channel', 'amt_src1', 'amt_src2', 'trans_type1', 'trans_type2']
    
    drop_train_dict = get_item_drop_list(op_train, trans_train, op_test, trans_test, op_drop_item, trans_drop_item)        #获得drop表（train中A榜没有的数据）
    
    #20181213 尝试drop train同分布
    op_train_deal, trans_train_deal = get_A_list_deal(op_train, trans_train, drop_train_dict, op_drop_item, trans_drop_item) 
    
    
    #最终处理,重新save
    op_train_deal.to_csv('../CleanedData/op_train_deal.csv', index = False)
    trans_train_deal.to_csv('../CleanedData/trans_train_deal.csv', index = False)
    op_test.to_csv('../CleanedData/op_test_deal.csv', index = False)
    trans_test.to_csv('../CleanedData/trans_test_deal.csv', index = False)
    label.to_csv('../CleanedData/tag_train_new.csv', index = False)
    sub.to_csv('../CleanedData/submit_example.csv', index = False)
    
#>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
    
if __name__ == "__main__":
    data_Cleaning_run()































