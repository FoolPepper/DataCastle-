# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import re
from Statistic import AllCateBadRateCount, mergetrainLabel
from drop_nor_record import muti_fea_drop_list, bb_drop_list_1,bb_drop_list_2,bb_drop_list_3,bb_drop_list_4
from collections import Counter



def drop_unimportant_col(fea_df):
    '''
    drop lgb模型给分低的特征
    
    '''
    
    importance_threshold = 10
    #important_fea = list(fea_importance_list)
    fea_importance_df_0 = pd.read_csv('../feature_2/fea_importance.csv', index_col=0)
    #fea_importance_df_1 = pd.read_csv('../fea_importance/muti_fea_importance_4.csv', index_col=0)
    #fea_importance_df_2 = pd.read_csv('../fea_importance/muti_fea_importance_5.csv', index_col=0)
    #fea_importance_df_3 = pd.read_csv('../fea_importance/all_fea_importance.csv', index_col=0)
    
    fea_unimportance_list_0 = list(fea_importance_df_0.loc[fea_importance_df_0['importance'] < importance_threshold]['column'])
    #fea_unimportance_list_1 = list(fea_importance_df_1.loc[fea_importance_df_1['importance'] < importance_threshold]['column'])
    #fea_unimportance_list_2 = list(fea_importance_df_2.loc[fea_importance_df_2['importance'] < importance_threshold]['column'])
    #fea_unimportance_list_3 = list(fea_importance_df_3.loc[fea_importance_df_3['importance'] < importance_threshold]['column'])
    
    fea_unimportance_list = []
    fea_unimportance_list.extend(fea_unimportance_list_0)
    #fea_unimportance_list.extend(fea_unimportance_list_1)
    #fea_unimportance_list.extend(fea_unimportance_list_2)
    #fea_unimportance_list.extend(fea_unimportance_list_3)
    
    fea_unimportance_list = list(set(fea_unimportance_list))
 
            
    for col in fea_df.columns:
        if col == 'Tag' or col == 'UID':
            continue
        if col  in fea_unimportance_list:
            fea_df = fea_df.drop([col], axis=1)
    
    orig_fea_num = len(fea_df.columns)
    drop_fea_num = len(fea_unimportance_list)
            
    return fea_df

def drop_col_KDE(fea_df):
    '''
    drop掉训练集与测试集 KDE 结果
    '''
    
#    drop_list = ['day0','day2','day3','day4','day5','day6','day7','day8',\
#              'day9','day10','day11','day12','day13','day14','day15',\
#              'day16','day17','day18','day19','day21','day22','day23',\
#              'day25','day26','day27','day28','day29','mode4','mode5',\
#              'mode6','mode8','mode9','mode10','mode25','mode30','mode52',\
#              'mode54','mode58','mode59','mode60','mode57','mode56','mode64','channel1','amt_src11']
#    drop_list = ['mode57','channel2','amt_src11','amt_src113','bal_mean','bal_max','mode26','mode29',\
#                 'channel_count_value', 'channel_entropy_value', 'amt_src1_entropy_value','trans_type1_entropy_value']
#    drop_list_baseline = ['trans_miss_max','bal_max','bal_mean','mode83','amt_src11','channel_y','day_y.2','code1_x',\
#             'code1_y','acc_id1_y','device_code1_x.1','device_code2_x.1','device1_x.1','device1_y.1','device2_x.1',\
#             'mac1_x.1','bal_y','bal_x.1','bal_y.1','bal_y.2','bal','market_code_y','market_type_y','market_type_x.2',\
#             ]
    
#    drop_list = ['amt_src11','channel_y','day_y.2','code1_x',\
#             'code1_y','acc_id1_y','device_code1_x.1','device_code2_x.1','device1_x.1','device1_y.1','device2_x.1',\
#             'mac1_x.1','bal_y','bal_x.1','bal_y.1','bal_y.2','bal','market_code_y','market_type_y','market_type_x.2',\
#             ]
    drop_list = bb_drop_list_4
    #drop_list.extend(bb_drop_list_4)
    #drop_list.extend(bb_drop_list_3)
    for col in list(fea_df.columns):
        if col in drop_list:
            fea_df = fea_df.drop([col], axis=1)
    
    return fea_df
    

def drop_day(fea_df):
    '''
    drop掉'day\d'特征
    '''
    match_list = list(fea_df.columns)
    
    for col in match_list:
        if re.search(r'day_\d',col) != None:
            fea_df = fea_df.drop([col], axis=1)
#    for col in match_list:
#        if re.search(r'day_\d_num',col) != None:
#            fea_df = fea_df.drop([col], axis=1)
    
#    for col in match_list:
#        if re.search(r'op_day_\d',col) != None:
#            fea_df = fea_df.drop([col], axis=1)
    
#    for col in match_list:
#        if re.search(r'op_day_\d_num',col) != None:
#            fea_df = fea_df.drop([col], axis=1)
    
    return fea_df


def col_normalize(df,col_list):
    
    
#    col_list = ['device_code1_y','device_code2_y','mac1_y','mac2_y','ip2_x','ip2_sub_x','ip2_sub_y','channel_x',\
#            'channel_x.2','day_x.1','day_x.3','time_x.1','time_y.1','trans_amt_x','trans_amt_x.1','trans_amt',\
#            'amt_src1_x','merchant_x','merchant_y','trans_type1_x','device_code1_y.1','device_code3_x.1',\
#            'device_code3_y.1','device2_y.1','ip1_y.1','bal_x','bal_x.2','acc_id2_y','acc_id3_y','trans_type2_x',\
#            'trans_type2_x.2','market_code_x','market_type_x','device_Num','deviceCode_Null_num','trans_num']
    
    for col in col_list:
        if(col in list(df.columns)):
            max_value = df[col].max()
            min_value = df[col].min()
            df[col] = df.apply(normalize,axis=1,args=(col,max_value,min_value))
        
    return df
def normalize(df, col, max_value, min_value):

    
    nor_col = (df[col] - min_value)/(max_value - min_value)
    
    return nor_col
    
def geo_decode_lat(orig_str):
    base_32 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',\
               'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    odd = 1
    lat = [-90, 90]
    lon = [-180,180]
    if isinstance(orig_str, float):
        return np.nan
    try:
        orig_str = list(orig_str)
    except:
        break_point = 1
        print('Error!')
    try:
        for char in orig_str:
            bits = base_32.index(char)
            for i in range(5):
                bit = int((bits >> (4-i) & 1))
                if odd & 1:
                    mid = (lon[0] + lon[1]) / 2
                    lon[1 - bit] = mid
                else:
                    mid = (lat[0] + lat[1]) / 2
                    lat[1 - bit] = mid
                odd ^= 1
    except ValueError:
        break_point = 2
    
    x = (lat[0] + lat[1]) / 2
    y = (lon[0] + lon[1]) / 2
    x, y = millier_convertion(x, y)
    
    return x


def geo_decode_lon(orig_str):
    base_32 = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'b', 'c', 'd', 'e', 'f', 'g',\
               'h', 'j', 'k', 'm', 'n', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
    
    odd = 1
    lat = [-90, 90]
    lon = [-180,180]
    if isinstance(orig_str, float):
        return np.nan
    try:
        orig_str = list(orig_str)
    except:
        break_point = 1
        print('Error!')
    try:
        for char in orig_str:
            bits = base_32.index(char)
            for i in range(5):
                bit = int((bits >> (4-i) & 1))
                if odd & 1:
                    mid = (lon[0] + lon[1]) / 2
                    lon[1 - bit] = mid
                else:
                    mid = (lat[0] + lat[1]) / 2
                    lat[1 - bit] = mid
                odd ^= 1
    except ValueError:
        break_point = 2
    
    x = (lat[0] + lat[1]) / 2
    y = (lon[0] + lon[1]) / 2
    x, y = millier_convertion(x, y)
    
    return y

def millier_convertion(lat, lon):
    L = 6381372 * np.pi * 2
    W = L
    H = L / 2
    mill = 2.3
    x = lon * np.pi /180
    y = lat * np.pi /180
    y = 1.25 * np.log(np.tan(0.25 * np.pi + 0.4 * y))
    x = (W / 2) + (W / (2 * np.pi)) * x
    y = (H / 2) - (H / (2 * mill)) * y
    
    return x/1000, y/1000


def surface_distance(p1,p2):
    '''
    p1: 坐标1 玮经度 [lat, lon]
    p2: 坐标2 玮经度 [lat, lon]
    
    
    '''
    R = 6300
    thita = np.arccos(np.cos(np.pi*p1[0]/180) * np.cos(np.pi*p2[0]/180) * 2 * np.cos(np.pi*(p1[1] - p2[1])/180) + np.sin(np.pi*p1[0]/180)*np.sin(np.pi*p2[0]/180))
    
    return R*thita


#def get_lat_lon(df):
#    a = pd.DataFrame()
#    a['aaa'],a['bbb'] = df['geo_code'].apply(geo_decode)
#    
#    return a
    
def get_all_class_index(op_train, trans_train, op_test, trans_test, op_col_list, trans_col_list):
    '''
    op_train : op表训练集 type:dataframe
    trans_train: trans表训练集 type:dataframe
    op_test : op表训练集 type:dataframe
    trans_test: trans表训练集 type:dataframe
    
    op_classes: op表中每个列的类别列表的字典 type:dic
    trans_classes: trans表中每个列的类别列表的字典 type:dic
    '''
    
    #op_col_list = []
    #trans_col_list = []
    
    op_classes = {}
    trans_classes = {}
    
    for col in op_col_list:
        class_set = []
        class_set_train = set(op_train[col])
        class_set_test = set(op_test[col])
        class_set.extend(list(class_set_train))
        class_set.extend(list(class_set_test))
        class_list = list(set(class_set))
        class_list = sorted(class_list)
        op_classes[col] = class_list
        
    for col in trans_col_list:
        class_set = []
        class_set_train = set(trans_train[col])
        class_set_test = set(trans_test[col])
        class_set.extend(list(class_set_train))
        class_set.extend(list(class_set_test))
        class_list = list(set(class_set))
        class_list = sorted(class_list)
        op_classes[col] = class_list
        
    return op_classes, trans_classes

def df_cut(df, groupkey):
    """cut一个df，使其根据groupKey(用于groupby的主键)的不同，来切分为不同的dataFrame，并存在一个字典中，cnt为计数（cut成多少个类别）"""
    dict_a = {}
    cnt = 0
    for name, group in df.groupby(groupkey):
        #print(name)
        #print(group)
        #print(type(group))
        group.index = range(len(group))
        dict_a[name] = group
        
        cnt = cnt + 1
    return dict_a, cnt


def cal_count_fea(serise_col):
    
    return serise_col.count()

def cal_unique_fea(serise_col):
    
    return serise_col.nunique()
    
def cal_mean_fea(serise_col):
    
    return serise_col.mean()

def cal_entropy_fea(serise_col, distri):
    '''
    serise_col :一列数据  type : Serise
    distri : 这列数据对应的 label == 1 时的分布 type: dic
    
    entropy : 信息熵
    '''
    entropy = 0
    op_num = serise_col.size
    for index in range(op_num):
        #prob = distri[str(serise_col[index])
        if serise_col[index] not in list(distri.index): #测试集中出现，训练集中没有的
            prob = 0
        else:
            prob = distri['rate'][serise_col[index]]
        entropy +=  -prob * np.log(prob + 0.00001)
    return entropy


def cal_ratio_fea(serise_col, class_set):
    '''
    serise_col :一列数据  type : Serise
    class_set : 这列数据对应的所有类别  type: list
    
    ratio_fea ： 类别占比
    '''
    
    ratio_fea_count = [0*x for x in range(len(class_set))]
    op_num = serise_col.size
    for index in range(op_num):
        if serise_col[index] not in class_set:    #测试集中出现，训练集中没有的
            continue
        else:
            ratio_fea_count[class_set.index(serise_col[index])] += 1
    
    ratio_fea = [x/(sum(ratio_fea_count)+ 0.00001) for x in ratio_fea_count]
    ratio_fea = np.array(ratio_fea)
    items = pd.DataFrame()
    for i in range(len(ratio_fea)):
        items[serise_col.name+str(i)] = pd.Series(ratio_fea[i])
    return items
        
        
    
def cal_rank_fea(serise_col):
    
    pass
    

def op2fea(user_op, op_distri_dic, op_class_dic):
    
    '''
    user_op :一个UID对应的所有op记录   type: dataframe
    stat_fea_col_flag ：需要计算特征的列名  type: list
    op_distri_dic: 每个op的列字段对应的label == 1 的分布 type: dic
    op_class_dic: 每个op的列字段对应的全部类别 type: dic
    
    item_df:一个UID对应的op特征，一行     type:dataframe
    '''
    
    item_df = pd.DataFrame()
    item_df['UID'] = [user_op['UID'][0]]
    
    stat_fea_col_flag = ['day','mode','os']               #需要提取特征的列名

    
    for col_name in stat_fea_col_flag:   #遍历需要计算统计特征的columns
        col_df = pd.DataFrame()
        col_df['UID'] = [user_op['UID'][0]]
        
        if user_op[col_name].dtype in ['int64', 'float64']:
            col_df[col_name+'_mean_value'] = pd.Series(cal_mean_fea(user_op[col_name]))
        col_df[col_name+'_count_value'] = pd.Series(cal_count_fea(user_op[col_name]))
        col_df[col_name+'_unique_value'] = pd.Series(cal_unique_fea(user_op[col_name]))
        col_df[col_name+'_entropy_value'] = pd.Series(cal_entropy_fea(user_op[col_name],op_distri_dic[col_name]))
        #col_df[col_name+'_ratio_value'] = pd.Series(cal_ratio_fea(user_op[col_name], op_class_dic[col_name]))
        #col_df[col_name+'rank_value'] = pd.Series(cal_rank_fea(user_op[col_name]))
        col_df = pd.concat([col_df, cal_ratio_fea(user_op[col_name], op_class_dic[col_name])],axis=1)
        item_df = item_df.merge(col_df,on='UID')
    
    
    return item_df

def trans2fea(user_trans, trans_distri_dic, trans_class_dic):
    
    '''
    user_trans :一个UID对应的所有交易记录 type: dataframe
    stat_fea_col_flag ：需要计算特征的列名 type: list
    trans_distri_dic: 每个trans的列字段对应的label == 1 的分布 type: dic
    trans_class_dic: 每个trans的列字段对应的全部类别 type: dic    
    
    item_df:一个UID对应的op特征,一行     type:dataframe
    '''
    
    item_df = pd.DataFrame()
    item_df['UID'] = [user_trans['UID'][0]]
    stat_fea_col_flag = ['channel','amt_src1','trans_type1']               #需要提取特征的列名
    for col_name in stat_fea_col_flag:   #遍历需要计算统计特征的columns
        col_df = pd.DataFrame()
        col_df['UID'] = [user_trans['UID'][0]]
        
        if user_trans[col_name].dtype in ['int64', 'float64']:
            col_df[col_name+'_mean_value'] = pd.Series(cal_mean_fea(user_trans[col_name]))
        col_df[col_name+'_count_value'] = pd.Series(cal_count_fea(user_trans[col_name]))
        col_df[col_name+'_unique_value'] = pd.Series(cal_unique_fea(user_trans[col_name]))
        col_df[col_name+'_entropy_value'] = pd.Series(cal_entropy_fea(user_trans[col_name], trans_distri_dic[col_name]))
        #col_df[col_name+'_ratio_value'] = pd.Series(cal_ratio_fea(user_trans[col_name], trans_class_dic[col_name]))
        #col_df[col_name+'rank_value'] = pd.Series(cal_rank_fea(user_tans[col_name]))
        col_df = pd.concat([col_df, cal_ratio_fea(user_trans[col_name], trans_class_dic[col_name])], axis=1)
        item_df = item_df.merge(col_df,on='UID')
    
    
    return item_df

def get_op_fea(op_df_dic, op_disri, op_classes):
    '''
    op_df_dic : key:UID,value:操作记录 type:dic
    
    op_fea_df: 所有样本的op特征 type:dataframe
    '''
    op_fea_df = pd.DataFrame()
    
    for key in op_df_dic:
        op_fea_df = op_fea_df.append(op2fea(op_df_dic[key], op_disri, op_classes))
        
    return op_fea_df

def get_trans_fea(trans_df_dic, trains_disri, trans_classes):
    '''
    op_df_dic : key:UID,value:交易记录 type:dic
    
    trans_fea_df: 所有样本的trans特征 type:dataframe
    '''
    trans_fea_df = pd.DataFrame()
    
    for key in trans_df_dic:
        trans_fea_df = trans_fea_df.append(trans2fea(trans_df_dic[key], trains_disri, trans_classes))
        
    return trans_fea_df


def op_miss_count(user_op):
    '''
    统计单个UID的op表缺失值
    '''
    
    op_miss = pd.DataFrame()
    op_miss['UID'] = [user_op['UID'][0]]
    op_miss['op_miss_rate'] = pd.Series([(user_op.size - sum(user_op.count()))/user_op.size])
    max_miss = 0
    for i in range(user_op.shape[0]):
        if max_miss < user_op.iloc[i].size - user_op.iloc[i].count():
            max_miss = user_op.iloc[i].size - user_op.iloc[i].count()
    op_miss['op_miss_max'] = pd.Series([max_miss])
    
    return op_miss
    
def trans_miss_count(user_trans):
    '''
    统计单个UID的trans表缺失值
    '''
    trans_miss = pd.DataFrame()
    trans_miss['UID'] = [user_trans['UID'][0]]
    trans_miss['trans_miss_rate'] = pd.Series([(user_trans.size - sum(user_trans.count()))/user_trans.size])
    max_miss = 0
    for i in range(user_trans.shape[0]):
        if max_miss < user_trans.iloc[i].size - user_trans.iloc[i].count():
            max_miss = user_trans.iloc[i].size - user_trans.iloc[i].count()
    trans_miss['trans_miss_max'] = pd.Series([max_miss])
    
    return trans_miss

def get_op_miss_fea(op_df_dic):
    op_miss_fea = pd.DataFrame()
    for key in op_df_dic:
        op_miss_fea = op_miss_fea.append(op_miss_count(op_df_dic[key]))
    
    return op_miss_fea

def get_trans_miss_fea(trans_df_dic):
    trans_miss_fea = pd.DataFrame()
    for key in trans_df_dic:
        trans_miss_fea = trans_miss_fea.append(trans_miss_count(trans_df_dic[key]))
    
    return trans_miss_fea

def cal_total_ave_miss_fea(total_miss_fea, op_item_num, trans_item_num):
    op_miss_fea = total_miss_fea['op_miss_rate']
    trans_miss_rate = total_miss_fea['trans_miss_rate']
    
    total_ave_miss_rate = (op_miss_fea*op_item_num + trans_miss_rate*trans_item_num) / (op_item_num + trans_item_num)
    
    return total_ave_miss_rate

def get_total_miss_fea(op_df_dic, trans_df_dic):
    op_miss_fea = get_op_miss_fea(op_df_dic)
    trans_miss_fea = get_trans_miss_fea(trans_df_dic)
    
    #op_item_num = op_miss_fea.columns.size - 1
    #trans_item_num = trans_miss_fea.columns.size - 1
    
    #total_miss_fea = pd.merge(op_miss_fea, trans_miss_fea, on='UID')
    
    #total_miss_fea['total_ave_miss_rate'] = total_miss_fea.apply(cal_total_ave_miss_fea, axis=1, args=(op_item_num, trans_item_num))
    
    #return total_miss_fea
    return op_miss_fea, trans_miss_fea
    

def trans_num2fea(user_trans):
    num_fea = pd.DataFrame()
    num_fea['UID'] = [user_trans['UID'][0]]
    
    stat_fea_col_flag = ['trans_amt','bal']               #需要提取特征的列名
    for col_name in stat_fea_col_flag:   #遍历需要计算统计特征的columns
        col_df = pd.DataFrame()
        col_df['UID'] = [user_trans['UID'][0]]
        col_df[col_name+'_max'] = pd.Series(max(user_trans[col_name]))
        col_df[col_name+'_mean'] = pd.Series(np.mean(user_trans[col_name]))
        
        num_fea = num_fea.merge(col_df, on='UID')
    
    return num_fea

def get_trans_num_fea(trans_df_dic):
    trans_num_fea = pd.DataFrame()
    for key in trans_df_dic:
        trans_num_fea = trans_num_fea.append(trans_num2fea(trans_df_dic[key]))
    
    return trans_num_fea

def region2fea(user_region):
    '''
    user_region : 一个UID对应的所有的region_label type:Dataframe
    
    region_fea: 一个UID对应的2个特征
    '''
    region_fea = pd.DataFrame()
    region_fea['UID'] = [user_region['UID'][0]]
    region_fea['unique_region'] = pd.Series(user_region['region_label'].nunique())
    region_fea['most_region_label'] = pd.Series(Counter(list(user_region['region_label'])).most_common(1)[0][0])
    
    return region_fea

def get_region_fea(df_dic):
    '''
    df_dic : key:UID  value:多条交易信息对应的region 
    
    region_fea： 每个UID对应的用多个region提取的特征
    '''
    region_fea = pd.DataFrame()
    for key in df_dic:
        region_fea = region_fea.append(region2fea(df_dic[key]))
    
    return region_fea

def get_all_region_fea(train_data, test_data):
    '''
    提取跟经纬度相关的特征
    train_data: 训练集的region type:Dataframe
    test_data:  测试集的region type:Dataframe
    
    '''
    train_df_dic, _ = df_cut(train_data, 'UID')
    test_df_dic, _ = df_cut(test_data, 'UID')
    
    train_region_fea = get_region_fea(train_df_dic)
    test_region_fea = get_region_fea(test_df_dic)
    
    return train_region_fea, test_region_fea
    
    
    

def get_all_fea_train(op_dir, trans_dir, label_dir):
    
    op_train = pd.read_csv(op_dir)
    trans_train = pd.read_csv(trans_dir)
    
    train_label = pd.read_csv(label_dir)
    
    
    op_df_dic, _ = df_cut(op_train, 'UID')
    trans_df_dic, _ = df_cut(trans_train, 'UID')
    
    
    op_strIndexs = ['day','mode','os']
    trans_strIndexs = ['channel','amt_src1','trans_type1']
    
    op_classes, trans_classes = get_all_class_index(op_train, trans_train, op_strIndexs, trans_strIndexs)
    
    op_train_lable, trans_train_lable = mergetrainLabel(op_train, trans_train, train_label)
    

    
    op_disri, trans_disri = AllCateBadRateCount(op_train_lable, trans_train_lable, op_strIndexs, trans_strIndexs)
    
    op_fea_df = get_op_fea(op_df_dic, op_disri, op_classes)
    trans_fea_df = get_trans_fea(trans_df_dic, trans_disri, trans_classes)
    
    
    #op_trans_fea_df = pd.merge(op_fea_df, trans_fea_df, how='inner', on='UID')

    
    op_miss_fea, trans_miss_fea = get_total_miss_fea(op_df_dic, trans_df_dic)
    
    trans_num_fea = get_trans_num_fea(trans_df_dic)
    
    return op_fea_df, trans_fea_df, op_miss_fea, trans_miss_fea, trans_num_fea


def get_all_fea_test(train_op_dir, train_trans_dir, test_op_dir, test_trans_dir, train_label_dir):
    '''
    op_dir: test_op表地址
    trains_dir: test_trans表地址
    train_label_dir: train_labe表地址
    
    提取测试集特征
    
    
    '''
    
    op_train = pd.read_csv(train_op_dir)
    trans_train = pd.read_csv(train_trans_dir)
    
    
    op_test = pd.read_csv(test_op_dir)
    trans_test = pd.read_csv(test_trans_dir)
    
    train_label = pd.read_csv(train_label_dir)
    
    
    op_df_dic, _ = df_cut(op_test, 'UID')
    trans_df_dic, _ = df_cut(trans_test, 'UID')
    
    
    op_strIndexs = ['day','mode','os']
    trans_strIndexs = ['channel','amt_src1','trans_type1']
    
    op_classes, trans_classes = get_all_class_index(op_train, trans_train, op_test, trans_test, op_strIndexs, trans_strIndexs)
    
    op_train_lable, trans_train_lable = mergetrainLabel(op_train, trans_train, train_label)
    

    
    op_disri, trans_disri = AllCateBadRateCount(op_train_lable, trans_train_lable, op_strIndexs, trans_strIndexs)
    
    op_fea_df = get_op_fea(op_df_dic, op_disri, op_classes)
    trans_fea_df = get_trans_fea(trans_df_dic, trans_disri, trans_classes)
    
    
    #op_trans_fea_df = pd.merge(op_fea_df, trans_fea_df, how='inner', on='UID')

    
    op_miss_fea, trans_miss_fea = get_total_miss_fea(op_df_dic, trans_df_dic)
    
    trans_num_fea = get_trans_num_fea(trans_df_dic)
    
    return op_fea_df, trans_fea_df, op_miss_fea, trans_miss_fea, trans_num_fea
    

if __name__ == '__main__':
#    op_train_dir = '../data/CleanedData/op_train_clean.csv'
#    trans_train_dir = '../data/CleanedData/trans_train_clean.csv'
#    op_test_dir = '../data/CleanedData/op_test_clean.csv'
#    trans_test_dir = '../data/CleanedData/trans_test_clean.csv'
#    label_dir = '../data/NewData/tag_train_new.csv'
#    
#    op_fea_df_train, trans_fea_df_train, op_miss_fea_train, trans_miss_fea_train, trans_num_fea_train =\
#    get_all_fea_train(op_train_dir, trans_train_dir, label_dir)
#    
#    op_fea_df_test, trans_fea_df_test, op_miss_fea_test, trans_miss_fea_test, trans_num_fea_test = \
#    get_all_fea_test(op_train_dir, trans_train_dir, op_test_dir, trans_test_dir, label_dir)
#    
#    op_fea_df_train.to_csv('../data1/op_fea_df_train.csv')
#    trans_fea_df_train.to_csv('../data1/trans_fea_df_train.csv')
#    op_miss_fea_train.to_csv('../data1/op_miss_fea_train.csv')
#    trans_miss_fea_train.to_csv('../data1/trans_miss_fea_train.csv')
#    trans_num_fea_train.to_csv('../data1/trans_num_fea_train.csv') 
#    
#    
#    op_fea_df_test.to_csv('../data1/op_fea_df_test.csv')
#    trans_fea_df_test.to_csv('../data1/trans_fea_df_test.csv')
#    op_miss_fea_test.to_csv('../data1/op_miss_fea_test.csv')
#    trans_miss_fea_test.to_csv('../data1/trans_miss_fea_test.csv')
#    trans_num_fea_test.to_csv('../data1/trans_num_fea_test.csv')
    
#    df = pd.read_csv('../data/CleanedData/trans_train_clean.csv')
#    lon = df['geo_code'].apply(geo_decode_lon)
#    lat = df['geo_code'].apply(geo_decode_lat)
#    df['x'] = lon
#    df['y'] = lat
    
    train_region = pd.read_csv('../data/train_region_label.csv',index_col=0)
    test_region = pd.read_csv('../data/test_region_label.csv',index_col=0)
    
    train_region_fea, test_region_fea = get_all_region_fea(train_region, test_region)
    
    train_region_fea.to_csv('../feature_1/train_region_fea.csv')
    test_region_fea.to_csv('../feature_1/test_region_fea.csv')
    