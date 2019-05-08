# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import data_preprocess

def get_feature(op,trans,label):
    for feature in op.columns[:]:
        if feature not in ['day']:
            if feature != 'UID':
                label = label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for cross_fea in ['ip1','mac1','mac2','geo_code']:
                if feature not in cross_fea:
                    if feature != 'UID':
                        temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp

        else:
            print(feature)
            label =label.merge(op.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(op.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for cross_fea in ['ip1','mac1','mac2']:
                if feature not in cross_fea:
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].max().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].min().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].sum().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].mean().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = op[['UID',cross_fea]].merge(op.groupby([cross_fea])[feature].std().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
                    
                    
    for feature in trans.columns[1:]:
        if feature not in ['trans_amt','bal','day']:
            if feature != 'UID':
                label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
                label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            for cross_fea in ['merchant','ip1','mac1','geo_code',]:
                if feature not in cross_fea: 
                    if feature != 'UID':
                        temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                        temp = temp.groupby('UID')[feature].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                    else:
                        temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp
                        temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID_x','UID_y']] 
                        temp = temp.groupby('UID_x')['UID_y'].sum().reset_index()
                        temp.columns = ['UID',feature+cross_fea]
                        label =label.merge(temp,on='UID',how='left')
                        del temp

        else:
            print(feature)
            label =label.merge(trans.groupby(['UID'])[feature].count().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].nunique().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].max().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].min().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].sum().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].mean().reset_index(),on='UID',how='left')
            label =label.merge(trans.groupby(['UID'])[feature].std().reset_index(),on='UID',how='left')
            for cross_fea in ['merchant','ip1','mac1']:
                if feature not in cross_fea:
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].count().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].nunique().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].sum().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].max().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].min().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].sum().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].mean().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    temp = trans[['UID',cross_fea]].merge(trans.groupby([cross_fea])[feature].std().reset_index(),on=cross_fea,how='left')[['UID',feature]] 
                    temp = temp.groupby('UID')[feature].mean().reset_index()
                    temp.columns = ['UID',feature+cross_fea]
                    label =label.merge(temp,on='UID',how='left')
                    del temp
                    
    print("Done")
    return label

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

def get_feature_1():
	op_train, trans_train, op_test, trans_test, y, sub = get_cleaned_data()
	train = get_feature(op_train,trans_train,y).fillna(-1)
	test = get_feature(op_test,trans_test,sub).fillna(-1)

	train.to_csv('../Feature/bb_fea_train_clean.csv',index=False)
	test.to_csv('../Feature/bb_fea_test_clean.csv',index=False)


#if __name__ == "__main__":
#    
#    get_feature_1(op_train, trans_train, op_test, trans_test, y, sub)
#	
	
	
	