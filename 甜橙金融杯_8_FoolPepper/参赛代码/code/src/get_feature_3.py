# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:33:52 2018

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

####################### >>>>>>> ##################
def median_in_list(_series):
    """list的中位数"""
    temp = np.array(_series.values[0])
    if np.isnan(temp).all():   #因为传过来的series的索引index是在原df中的索引，所以取values，然后用[0]取值.all用于集合的真假
        return np.nan
    else:
        return np.median(_series.values[0])
    
def count_rate_in_list(_series, Num = 1):
    """小于等于Num的比率,此函数可以计算更多的可能,题目中用于计算1的比率"""
    temp = np.array(_series.values[0])
    if np.isnan(temp).all():   #因为传过来的series的索引index是在原df中的索引，所以取values，然后用[0]取值.all用于集合的真假
        return np.nan
    else:
        bool_array = (temp <= Num)           #对小于等于1的计数，这个可以对小于更多的东西计数,Num为小于等于的数
        return (np.sum(bool_array))/len(temp)        #比率

def get_dict_count(_series, _dict):
    """功能：返回在group中，的_fea对应的原始数据（如_fea = ip1）,返回上过此IP的uid_num的数量（返回一个列表）"""
    """_series为feature对应的一列数据，_df为原始df, _fea为feature的名字"""
    fea_cate = _series.unique()
    uid_num = [_dict[cate] for cate in fea_cate if cate is not np.nan]   #在——df中求此_fea对应几个不同的UID,得到其数量

    return uid_num

#    
def get_feabook_UIDNum(df, fea_str):
    """用于寻找IP对应的UID数量,返回一个Series，当做字典用"""
    UIDNum_book = df.groupby(df[fea_str])['UID'].nunique()
    
    UIDNum_dict = UIDNum_book.to_dict()
    return UIDNum_dict
#
def get_feabook_feaNum(df, fea_str):
    """用于计算fea的数量,返回一个Series，当做字典用"""
    FeaCount_book = df.groupby(df[fea_str])['UID'].count()
    FeaCount_dict = FeaCount_book.to_dict()
    return FeaCount_dict

def crossCount(df, fea_str):
    """功能：UID->fea->UID_NUM, df为原始df, fea为特征名称"""
    print('cross start:' + fea_str); print("running...")
    
    UIDNum_dict = get_feabook_UIDNum(df, fea_str)       #获得查询表
    FeaNum_dict = get_feabook_feaNum(df, fea_str)
    
    df_sub = df[['UID', fea_str]]
    temp = df_sub.groupby(df_sub['UID'])[fea_str].apply(get_dict_count, _dict = UIDNum_dict)   
    temp = temp.reset_index().rename(columns = {fea_str: 'cate_list'})
    
    temp2 = df_sub.groupby(df_sub['UID'])[fea_str].apply(get_dict_count, _dict = FeaNum_dict)
    temp2 = temp2.reset_index().rename(columns = {fea_str: 'fea_num_list'})
    """返回一个df, UID, cate_list(存放为UID出现的ip, 在上面登陆过的UID的数目)"""
    temp = pd.merge(temp, temp2, on = 'UID', how = 'left')
    return temp

def getCrossFeature(cross_origin_fea_df, fea_str, df_name):
    median_df = cross_origin_fea_df.groupby(cross_origin_fea_df['UID'])['cate_list'].apply(lambda x:median_in_list(x)).reset_index().rename(columns = {'cate_list': fea_str + '_median_in_list_' + df_name})
    rate_1_df = cross_origin_fea_df.groupby(cross_origin_fea_df['UID'])['cate_list'].apply(count_rate_in_list, Num = 1).reset_index().rename(columns = {'cate_list':fea_str + '_1_rate_in_list_' + df_name})
    ########    
    cross_origin_fea_df = pd.merge(cross_origin_fea_df, median_df, on = 'UID', how = 'left')
    cross_origin_fea_df = pd.merge(cross_origin_fea_df, rate_1_df, on = 'UID', how = 'left')

    return cross_origin_fea_df

def return_cross_feature(df, df_type, fea_str, label):
    """返回df(原始df), df_type('trans'或者'op')字符串，fea_str特征名称字符串"""
    print("Now start calculate cross feature...")
    print("type is " + df_type + " feature is " + fea_str)
    
    t1 = time.time()
    print("get list...")
#    op_train_sub = crossCount(op_train, 'ip1')
#    op_train_sub = pd.merge(label, op_train_sub, on = 'UID', how = 'left')
#    print("get list done")
#    
#    op_train_sub = getCrossFeature(op_train_sub, 'ip1', 'op')

    cross_feature = crossCount(df, fea_str)
    cross_feature = pd.merge(label, cross_feature, on = 'UID', how = 'left')
    print("get list done")
    
    cross_feature = getCrossFeature(cross_feature, fea_str, df_type)
#
    t2 = time.time()
    print("done,time count is ", t2 - t1)
    print('______________________________')
    
    #drop掉多余数据
    cross_feature = cross_feature.drop(columns = ['Tag', 'cate_list', 'fea_num_list'])
    return cross_feature

def get_ip1_feature(op_train, trans_train, op_test, trans_test, label, sub):
    """ip1 feature(由于ip1的特殊性，取中位数和1的比率)"""
    train_cross_feature_output = label
    test_cross_feature_output = sub
    #成为母Df
    train_cross_feature_output = train_cross_feature_output.drop(columns = ['Tag'])
    test_cross_feature_output = test_cross_feature_output.drop(columns = ['Tag'])

    op_feature_list = ['ip1']
    
    #train: trans, op
    print('start caluculate cross feature!')
    for index in op_feature_list:
        cross_feature = return_cross_feature(op_train, 'op', index, label)
        train_cross_feature_output = pd.merge(train_cross_feature_output, cross_feature, on = 'UID', how = 'left')
        #print(cross_feature)
    print('train done')
    
    #test: trans, op
    for index in op_feature_list:
        cross_feature = return_cross_feature(op_test, 'op', index, sub)
        test_cross_feature_output = pd.merge(test_cross_feature_output, cross_feature, on = 'UID', how = 'left')
        #print(cross_feature)        
    print('test done!')
    print('done!')
    
    train_cross_feature_output.to_csv('../Feature/ip1_feature_train.csv', index = False)
    test_cross_feature_output.to_csv('../Feature/ip1_feature_test.csv', index = False)

def get_feature_3():
    op_train, trans_train, op_test, trans_test, label, sub = get_cleaned_data()   
    get_ip1_feature(op_train, trans_train, op_test, trans_test, label, sub)

    
if __name__ == "__main__":  
    get_feature_3()