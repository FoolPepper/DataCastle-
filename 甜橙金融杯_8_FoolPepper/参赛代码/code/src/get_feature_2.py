# -*- coding: utf-8 -*-
"""
Created on Tue Nov 13 08:36:34 2018

@author: FoolPepper
"""

import pandas as pd
import time

import config
import logging
import warnings

from data_preprocess import data_preprocess
from get_preFeature import get_preFeature


warnings.filterwarnings('ignore')

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )

def load_data(data_type="train"):
    logging.info("loading {} data...".format(data_type))
    time_point = time.time()
    if data_type == "train":
        opera = pd.read_csv(config.NEW_RAW_FEATURE_OPERA_TRAIN, low_memory=False)
        transac = pd.read_csv(config.NEW_RAW_FEATURE_TRANSAC_TRAIN, low_memory=False)
    else:
        opera = pd.read_csv(config.NEW_RAW_FEATURE_OPERA_TEST, low_memory=False)
        transac = pd.read_csv(config.NEW_RAW_FEATURE_TRANSAC_TEST, low_memory=False)
        
    logging.info("loading done, cost time {}s".format(time.time()-time_point))
    
    return opera, transac

    

def opera_feature(opera, data):
    
    # feature_unique, count
    for feature in config.OPERA_FEATURE_OBJECT:
        logging.info("{0}_unique_opera...".format(feature))
        temp = opera.groupby(["UID"], as_index=False)[feature].agg({"{}_count_opera".format(feature):"count", 
                                                                    "{}_unique_opera".format(feature):"nunique"})
        data = pd.merge(data, temp, how="left", on=["UID"])


    for feature in config.OPERA_FEATURE_OBJECT_UNIQUE:
        logging.info("{0}_unique_opera...".format(feature))
        temp = opera.groupby(["UID"], as_index=False)[feature].agg({"{}_unique_opera".format(feature):"nunique"})
        data = pd.merge(data, temp, how="left", on=["UID"])

    # 统计一些数值特征
    for item in config.OPERA_FEATURE_NOT_OBJECT_MEAN:
        logging.info("{}_feature_opera...".format(item))
        temp = opera.groupby(["UID"], as_index=False)[item].agg({"{}_mean_opera".format(item):"mean"})
        data = pd.merge(data, temp, how="left", on=["UID"])

    return data


def transac_feature(transac, data):

    # feature_unique...
    for item in config.TRANSAC_FEATURE_OBJECT:
        logging.info("{0}_unique_transac...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_count_transac".format(item):"count", 
                                                                   "{}_unique_transac".format(item):"nunique"})
        data = pd.merge(data, temp, how="left", on=["UID"])

    for item in config.TRANSAC_FEATURE_OBJECT_UNIQUE:
        logging.info("{0}_unique_transac...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_unique_transac".format(item):"nunique"})
        data = pd.merge(data, temp, how="left", on=["UID"])

   
    for item in config.TRANSAC_FEATURE_NOT_OBJECT_MEAN:
        logging.info("{}_feature_transac...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_mean_transac".format(item):"mean"})
        data = pd.merge(data, temp, how="left", on=["UID"])
    
    for item in config.TRANSAC_FEATURE_NOT_OBJECT:
        logging.info("{}_feature_transac...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_max_transac".format(item):"max"})
        data = pd.merge(data, temp, how="left", on=["UID"])
    
    for item in config.TRANSAC_FEATURE_NOT_OBJECT_MIN:
        logging.info("{}_feature_transac...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_min_transac".format(item):"min"})
        data = pd.merge(data, temp, how="left", on=["UID"])
    
    for item in config.TRANSAC_FEATURE_NOT_OBJECT_STD:
        logging.info("{}_feature_transac_std...".format(item))
        temp = transac.groupby(["UID"], as_index=False)[item].agg({"{}_std_transac".format(item):"std"})
        data = pd.merge(data, temp, how="left", on=["UID"])

        
    return data


def filter_the_feature(data):
    
    filter_feature = config.FILTER_FEATURE + ["UID"]
    data = data[filter_feature]
    
    return data


def get_feature_2():
    logging.info("data_preprocess...")      #数据预处理
    data_preprocess()
    
    logging.info("prefeature...")
    get_preFeature()                        #中间结果处理
    
    logging.info("loading...")
    opera_train, transac_train = load_data("train")
    opera_test, transac_test = load_data("test")
    
    
    
    train_data = pd.read_csv(config.RAW_DATA_LABEL_TRAIN, low_memory=False)
    test_data = pd.read_csv(config.RAW_DATA_LABEL_TEST, low_memory=False)
    
    logging.info("transac_feature...")
    train_data = transac_feature(transac_train, train_data)
    test_data = transac_feature(transac_test, test_data)

    logging.info("opera_feature...")
    train_data = opera_feature(opera_train, train_data)
    test_data = opera_feature(opera_test, test_data)
    
    logging.info("filter...")
    train_data = filter_the_feature(train_data)
    test_data = filter_the_feature(test_data)
       
    logging.info("transac data to csv files...")
    train_data.fillna(-1)
    test_data.fillna(-1)
    train_data.to_csv(config.FEATURE_TRAIN, index=False)
    test_data.to_csv(config.FEATURE_TEST, index=False)
    
    logging.info("done")


if __name__ == "__main__":  
    get_feature_2()








