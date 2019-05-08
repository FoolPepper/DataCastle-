# -*- coding: utf-8 -*-
"""
Created on Sun Nov 11 15:00:25 2018

@author: FoolPepper
"""

import pandas as pd
import time

import config
import logging
import warnings

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
        opera = pd.read_csv(config.PRE_DATA_OPERA_TRAIN, low_memory=False)
        transac = pd.read_csv(config.PRE_DATA_TRANSAC_TRAIN, low_memory=False)
    else:
        opera = pd.read_csv(config.PRE_DATA_OPERA_TEST, low_memory=False)
        transac = pd.read_csv(config.PRE_DATA_TRANSAC_TEST, low_memory=False)
        
    logging.info("loading done, cost time {}s".format(time.time()-time_point))
    
    return opera, transac

def opera_feature(opera):
    
    #查看一下黑号都是在哪几天大量操作
    logging.info("mode times...")
    temp = opera.groupby(["UID", "new_day"], as_index=False)["mode"].agg({"mode_unique":"count"})
    opera = pd.merge(opera, temp, how="left", on=["UID", "new_day"])
    
    #同一设备识别码有多少用户 
    logging.info("device_code1 对应的多少用户")
    temp = opera.groupby(["device_code1"], as_index=False)["UID"].agg({"device1_UID_unique":"nunique"})
    temp = temp[temp.device_code1 != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["device_code1"])
    
    logging.info("device_code2 对应的多少用户")
    temp = opera.groupby(["device_code2"], as_index=False)["UID"].agg({"device2_UID_unique":"nunique"})
    temp = temp[temp.device_code2 != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["device_code2"])
    
    logging.info("device_code3 对应的多少用户")
    temp = opera.groupby(["device_code3"], as_index=False)["UID"].agg({"device3_UID_unique":"nunique"})
    temp = temp[temp.device_code3 != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["device_code3"])
    
    #在同一wifi下多个设备存在
    logging.info("wifi_device1_unique...")
    temp = opera.groupby(["wifi"], as_index=False)["device_code1"].agg({"wifi_device1_unique":"nunique"})
    temp = temp[temp.wifi != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["wifi"])

    logging.info("wifi_device2_unique...")
    temp = opera.groupby(["wifi"], as_index=False)["device_code2"].agg({"wifi_device2_unique":"nunique"})
    temp = temp[temp.wifi != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["wifi"])
    
    logging.info("wifi_device3_unique...")
    temp = opera.groupby(["wifi"], as_index=False)["device_code3"].agg({"wifi_device3_unique":"nunique"})
    temp = temp[temp.wifi != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["wifi"])
    
    #同一wifi下地理位置不同
    logging.info("wifi_geo_code_unique...")
    temp = opera.groupby(["wifi"], as_index=False)["geo_code"].agg({"wifi_geo_code_unique":"nunique"})
    temp = temp[temp.wifi != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["wifi"])
    
    #同一wifi下mac2数量
    logging.info("wifi_mac2_unique...")
    temp = opera.groupby(["wifi"], as_index=False)["mac2"].agg({"wifi_mac2_unique":"nunique"})
    temp = temp[temp.wifi != -1]
    opera = pd.merge(opera, temp, how="left", on=["wifi"])
    
    #同一ip2下，多少账户存在
    logging.info("ip2_UID_unique...")
    temp = opera.groupby(["ip2"], as_index=False)["UID"].agg({"ip2_UID_unique":"nunique"})
    temp = temp[temp.ip2 != -1]
    opera = pd.merge(opera, temp, how="left", on=["ip2"])
    
    #同一小时内，多少账户进行相同的操作

    
    #同一小时内，相同设备登录多少不同账户
    logging.info("hour_device_code1_UID_unique")
    temp = opera.loc[(opera.Hour!=-1)&(opera.device_code1!=-1)].groupby(["Hour", "device_code1"], as_index=False)["UID"].agg({"hour_device_code1_UID_unique":"nunique"})
    opera = pd.merge(opera, temp, how="left", on=["Hour", "device_code1"])

    #同一设备mac1地址下多少用户存在
    logging.info("mac1_UID_unique")
    temp = opera.groupby(["mac1"], as_index=False)["UID"].agg({"mac1_UID_unique":"nunique"})
    temp = temp[temp.mac1 != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["mac1"])

    #同一mac2地址下多少用户存在
    logging.info("mac2_UID_unique")
    temp = opera.groupby(["mac2"], as_index=False)["UID"].agg({"mac2_UID_unique":"nunique"})
    temp = temp[temp.mac2 != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["mac2"])
    
    #设备ip1_sub下用户数量
    logging.info("ip1_sub_UID_unique")
    temp = opera.groupby(["ip1_sub"], as_index=False)["UID"].agg({"ip1_sub_UID_unique":"nunique"})
    temp = temp[temp.ip1_sub != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["ip1_sub"])
    
    #设备ip2_sub下用户数量    
    logging.info("ip2_sub_UID_unique")
    temp = opera.groupby(["ip2_sub"], as_index=False)["UID"].agg({"ip2_sub_UID_unique":"nunique"})
    temp = temp[temp.ip2_sub != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["ip2_sub"])
    
    #同一地理区域下的device1设备数量
    logging.info("geo_code_device1_unique")
    temp = opera.groupby(["geo_code"], as_index=False)["device_code1"].agg({"geo_code_device1_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["geo_code"])
    
    #同一地理区域下的device2设备数量
    logging.info("geo_code_device2_unique")
    temp = opera.groupby(["geo_code"], as_index=False)["device_code2"].agg({"geo_code_device2_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["geo_code"])
    
    #同一地理区域下的device3设备数量
    logging.info("geo_code_device3_unique")
    temp = opera.groupby(["geo_code"], as_index=False)["device_code3"].agg({"geo_code_device3_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["geo_code"])

    #同一地理区域下的用户数量
    logging.info("geo_code_UID_unique")
    temp = opera.groupby(["geo_code"], as_index=False)["UID"].agg({"geo_code_UID_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    opera = pd.merge(opera, temp, how="left", on=["geo_code"])
    
    #同一地理区域及wifi下用户数量
    logging.info("geo_code_wifi_UID_unique")
    temp = opera[(opera.geo_code!=-1) & (opera.wifi!=-1)].groupby(["geo_code", "wifi"], as_index=False)["UID"].agg({"geo_code_wifi_UID_unique":"nunique"})
    opera = pd.merge(opera, temp, how="left", on=["geo_code", "wifi"])
    
    #同一地理区域及wifi下device_code1数量
    logging.info("geo_code_wifi_device_code1_unique")
    temp = opera[(opera.geo_code!=-1) & (opera.wifi!=-1)].groupby(["geo_code", "wifi"], as_index=False)["device_code1"].agg({"geo_code_wifi_device_code1_unique":"nunique"})
    opera = pd.merge(opera, temp, how="left", on=["geo_code", "wifi"])

    #同一地理区域及wifi下device_code2数量
    logging.info("geo_code_wifi_device_code2_unique")
    temp = opera[(opera.geo_code!=-1) & (opera.wifi!=-1)].groupby(["geo_code", "wifi"], as_index=False)["device_code2"].agg({"geo_code_wifi_device_code2_unique":"nunique"})
    opera = pd.merge(opera, temp, how="left", on=["geo_code", "wifi"])

    #同一地理区域及wifi下device_code3数量
    logging.info("geo_code_wifi_device_code3_unique")
    temp = opera[(opera.geo_code!=-1) & (opera.wifi!=-1)].groupby(["geo_code", "wifi"], as_index=False)["device_code3"].agg({"geo_code_wifi_device_code3_unique":"nunique"})
    opera = pd.merge(opera, temp, how="left", on=["geo_code", "wifi"])

    #同一动作，同一用户操作多少次
    logging.info("mode_UID_count")
    temp = opera.groupby(["mode", "UID"], as_index=False)["Tag"].agg({"mode_UID_count":"count"})
    opera = pd.merge(opera, temp, how="left", on=["mode", "UID"])
    
    #同一动作，同一用户成功操作多少次
    logging.info("mode_UID_success_count")
    temp = opera[opera.success!=-1].groupby(["mode", "UID"], as_index=False)["success"].agg({"mode_UID_success_count":"sum"})
    opera = pd.merge(opera, temp, how="left", on=["mode", "UID"])

    #不同时间细粒程度的刻画
    #同一小时内，同一用户操作多少次
    logging.info("day_Hour_mode_UID_count")
    temp =opera.groupby(["new_day", "Hour", "mode", "UID"], as_index=False)["Tag"].agg({"Hour_mode_UID_count":"count"})
    opera = pd.merge(opera, temp, on=["new_day", "Hour", "mode", "UID"], how="left")
    
    #同一小时内，同一用户成功操作多少次
    logging.info("day_Hour_mode_UID_success_count")
    temp =opera.groupby(["new_day", "Hour", "mode", "UID"], as_index=False)["success"].agg({"Hour_mode_UID_success_count":"sum"})
    opera = pd.merge(opera, temp, on=["new_day", "Hour", "mode", "UID"], how="left")
    
    
    ####关于day和
    for item in ["os", "version", "device1", "device2", 
                         "device_code1", "device_code2", "device_code3", 
                         "mac1", "ip1", "ip2", "mac2", "ip1_sub",
                         "ip2_sub"]:
        logging.info("day_UID_{}_unique".format(item))
        temp = opera.groupby(["new_day", "UID"], as_index=False)[item].agg({"day_UID_{}_unique".format(item):"nunique"})
        opera = pd.merge(opera, temp, how="left", on=["new_day", "UID"])
        
    #同一天，同一小时内 UID 出现在不同item特征...
    for item in ["os", "version", "device1", "device2", 
                              "device_code1", "device_code2", "device_code3", 
                              "mac1", "ip1", "ip2", "mac2", "ip1_sub",
                              "ip2_sub"]:
        logging.info("day_hour_UID_{}_unique".format(item))
        temp = opera.groupby(["new_day", "Hour", "UID"], as_index=False)[item].agg({"day_hour_UID_{}_unique".format(item):"nunique"})
        opera = pd.merge(opera, temp, how="left", on=["new_day", "Hour", "UID"])
    
    #同一天，同一小时内同一ip,或mac...上出现的不同UID数量
    for item in ["device1", "device2", "mac1", "device_code1", 
                               "device_code2", "device_code3", "ip1", "ip2", 
                               "mac2", "ip1_sub", "ip2_sub", "geo_code"]:
        logging.info("day_hour_{}_UID_unique".format(item))
        temp = opera[opera[item]!=-1].groupby(["new_day", "Hour", item], as_index=False)["UID"].agg({"day_hour_{}_UID_unique".format(item):"nunique"})
        opera = pd.merge(opera, temp, how="left", on=["new_day", "Hour", item])    
    
    #同一天，同一小时同一设备切换不同地区
    for item in ["os", "version", "device1", "device2", 
                              "device_code1", "device_code2", "device_code3", 
                              "mac1", "ip1", "ip2", "mac2", "ip1_sub",
                              "ip2_sub"]:
        logging.info("day_hour_{}_geo_code_unique".format(item))
        temp = opera[opera["geo_code"]!=-1].groupby(["new_day", "Hour", item], as_index=False)["geo_code"].agg({"day_hour_{}_geo_code_unique".format(item):"nunique"})
        opera = pd.merge(opera, temp, how="left", on=["new_day", "Hour", item])

    #同一天，同一设备切换不同地区
    for item in ["os", "version", "device1", "device2", 
                              "device_code1", "device_code2", "device_code3", 
                              "mac1", "ip1", "ip2", "mac2", "ip1_sub",
                              "ip2_sub"]:
        logging.info("day_{}_geo_code_unique".format(item))
        temp = opera[opera["geo_code"]!=-1].groupby(["new_day", item], as_index=False)["geo_code"].agg({"day_{}_geo_code_unique".format(item):"nunique"})
        opera = pd.merge(opera, temp, how="left", on=["new_day", item])

  
    return opera

def transac_feature(transac):
    # 对trans_amt 进行卡方分箱
    
    #同一设备识别码有多少用户
    logging.info("device_type count...")
    temp = transac.groupby(["UID"], as_index=False)["device_type"].agg({"device_type_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["UID"])
    
    logging.info("device_code1 对应的多少用户")
    temp = transac.groupby(["device_code1"], as_index=False)["UID"].agg({"device1_UID_unique":"nunique"})
    temp = temp[temp.device_code1 != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["device_code1"])
    
    logging.info("device_code2 对应的多少用户")
    temp = transac.groupby(["device_code2"], as_index=False)["UID"].agg({"device2_UID_unique":"nunique"})
    temp = temp[temp.device_code2 != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["device_code2"])
    
    logging.info("device_code3 对应的多少用户")
    temp = transac.groupby(["device_code3"], as_index=False)["UID"].agg({"device3_UID_unique":"nunique"})
    temp = temp[temp.device_code3 != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["device_code3"])
    
    #同一营销活动编码对应的UID数量
    logging.info("logging market_code_transac...")
    temp = transac[transac.market_code!=-1].groupby(["market_code"], as_index=False)["UID"].agg({"market_code_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["market_code"])
    
    #同一营销活动标识market_type对应的UID数量
    logging.info("logging market_type_transac...")
    temp = transac[transac.market_type!=-1].groupby(["market_type"], as_index=False)["UID"].agg({"market_type_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["market_type"])

    #同一营销活动标识market_type下相同mac2的UID数量
    logging.info("logging market_type_transac...")
    temp = transac[(transac.market_type!=-1) & (transac.mac1!=-1)].groupby(["market_type", "mac1"], as_index=False)["UID"].agg({"market_type_mac1_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["market_type", "mac1"])

    #同一营销活动标识market_type下相同acc_id3的UID数量
    logging.info("logging market_type_transac...")
    temp = transac[(transac.market_type!=-1) & (transac.acc_id3!=-1)].groupby(["market_type", "acc_id3"], as_index=False)["UID"].agg({"market_type_acc_id3_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["market_type", "acc_id3"])

    #同一mac1地址相同acc_id3的UID数量
    logging.info("mac1_acc_id3_UID")
    temp = transac[(transac.mac1!=-1)&(transac.acc_id3!=-1)].groupby(["mac1", "acc_id3"], as_index=False)["UID"].agg({"mac1_acc_id3_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["mac1", "acc_id3"])

    #同一设备mac1地址下多少用户存在
    logging.info("mac1_UID_unique")
    temp = transac.groupby(["mac1"], as_index=False)["UID"].agg({"mac1_UID_unique":"nunique"})
    temp = temp[temp.mac1 != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["mac1"])
    
    #设备ip1_sub下用户数量
    logging.info("ip1_sub_UID_count")
    temp = transac.groupby(["ip1"], as_index=False)["UID"].agg({"ip1_UID_unique":"nunique"})
    temp = temp[temp.ip1 != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["ip1"])

    #同一ip1_sub acc_id3的UID数量
    logging.info("ip1_sub_acc_id3_UID_count")
    temp = transac[(transac.ip1_sub!=-1)&(transac.acc_id3!=-1)].groupby(["ip1_sub", "acc_id3"], as_index=False)["UID"].agg({"ip1_sub_acc_id3_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, how="left", on=["ip1_sub", "acc_id3"])
        
    #同一地理区域下的device1设备数量
    logging.info("geo_code_device1")
    temp = transac.groupby(["geo_code"], as_index=False)["device_code1"].agg({"geo_code_device1_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["geo_code"])
    
    #同一地理区域下的device2设备数量
    logging.info("geo_code_device2")
    temp = transac.groupby(["geo_code"], as_index=False)["device_code2"].agg({"geo_code_device2_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["geo_code"])
    
    #同一地理区域下的device3设备数量
    logging.info("geo_code_device3")
    temp = transac.groupby(["geo_code"], as_index=False)["device_code3"].agg({"geo_code_device3_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["geo_code"])

    #同一地理区域下的用户数量
    logging.info("geo_code_UID")
    temp = transac.groupby(["geo_code"], as_index=False)["UID"].agg({"geo_code_UID_unique":"nunique"})
    temp = temp[temp.geo_code != "-1"]
    transac = pd.merge(transac, temp, how="left", on=["geo_code"])
    
    
    #当前小时内，每个item有多少个UID
    for item in config.TRANSAC_DAY_HOUR_UID:
        logging.info("day_hour_{}_UID_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["UID"].agg({"day_hour_{}_UID_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])
        
    #当前小时内，每个商户出现的不同item
    for item in config.TRANSAC_DAY_HOUR_MERCHANT:
        logging.info("day_hour_merchant_{}_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", "merchant"], as_index=False)[item].agg({"day_hour_merchant_{}_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", "merchant"])

    #当前小时内，每个子商户出现的不同item
    for item in config.TRANSAC_DAY_HOUR_MERCHANT:
        logging.info("day_hour_code1_{}_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", "code1"], as_index=False)[item].agg({"day_hour_code1_{}_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", "code1"])
    
    #当前小时内，每个商户终端出现的不同item
    for item in config.TRANSAC_DAY_HOUR_MERCHANT:
        logging.info("day_hour_code2_{}_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", "code2"], as_index=False)[item].agg({"day_hour_code2_{}_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", "code2"])

    #当前小时内，每个转入账户出现不同的item
    for item in config.TRANSAC_DAY_HOUR_ACC:
        logging.info("day_hour_acc_id3_{}_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", "acc_id3"], as_index=False)[item].agg({"day_hour_acc_id3_{}_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", "acc_id3"])
   
    
    #当前小时内，每个营销活动编码中出现不同的item
    for item in config.TRANSAC_DAY_HOUR_ACC:
        logging.info("day_hour_market_code_{}_unique".format(item))
        temp = transac.groupby(["new_day", "Hour", "market_code"], as_index=False)[item].agg({"day_hour_market_code_{}_unique".format(item):"nunique"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", "market_code"])

    
    #当前小时内，当前商户，子商户，商户终端设备，商户账户交易金额
    for item in ["merchant", "code1", "code2"]:
        logging.info("day_hour_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["trans_amt"].agg({"day_hour_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])
    
    #当前小时内，当前用户，设备，ip地址交易金额
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1"]:
        logging.info("day_hour_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["trans_amt"].agg({"day_hour_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])
 
    #当前小时内，当前用户，设备，ip地址交易次数
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1", "code2"]:
        logging.info("day_hour_{}_trans_amt_count".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["trans_amt"].agg({"day_hour_market_{}_trans_amt_count".format(item):"count"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])

#
    #当天内，当前商户，子商户，商户终端设备，商户账户交易金额
    for item in ["merchant", "code1", "code2"]:
        logging.info("day_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", item], as_index=False)["trans_amt"].agg({"day_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", item])
    
    #当天内，当前用户，设备，ip地址交易金额
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1"]:
        logging.info("day_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", item], as_index=False)["trans_amt"].agg({"day_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", item])
 
    #当天内，当前用户，设备，ip地址交易次数
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1", "code2"]:
        logging.info("day_{}_trans_amt_count".format(item))
        temp = transac.groupby(["new_day", item], as_index=False)["trans_amt"].agg({"day_market_{}_trans_amt_count".format(item):"count"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", item])

##
    #当前三小时内，当前商户
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1", "code2"]:
        logging.info("day_three_hour_{}_trans_amt_count".format(item))
        temp = transac.groupby(["new_day", "three_hour", item], as_index=False)["trans_amt"].agg({"day_three_hour_market_{}_trans_amt_count".format(item):"count"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "three_hour", item])

    #当前三小时，当前商户，子商户，商户终端设备，商户账户交易金额
    for item in ["merchant", "code1", "code2"]:
        logging.info("day_three_hour_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", "three_hour", item], as_index=False)["trans_amt"].agg({"day_three_hour_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "three_hour", item])
    
    #当前三小时内，当前用户，设备，ip地址交易金额
    for item in ["UID", "acc_id1", "device_code1", "device_code2", "device_code3", "mac1", 
                 "ip1_sub", "ip1"]:
        logging.info("day_three_hour_{}_trans_amt_sum".format(item))
        temp = transac.groupby(["new_day", "three_hour", item], as_index=False)["trans_amt"].agg({"day_three_hour_market_{}_trans_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "three_hour", item])

###    
    #当前一天内，在同一WiFi,mac2,os,ip2下交易的次数
    for item in ["wifi", "mac2", "os", "ip2"]:
        logging.info("day_{}_count".format(item))
        temp = transac.groupby(["new_day", item], as_index=False)["trans_amt"].agg({"day_{}_count".format(item):"count"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", item])
    
    #当前一天内，在同一WiFi,mac2,os,ip2下交易的金额
    for item in ["wifi", "mac2", "os", "ip2"]:
        logging.info("day_{}_transa_amt_sum".format(item))
        temp = transac.groupby(["new_day", item], as_index=False)["trans_amt"].agg({"day_{}_transa_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", item])
   
    #当前一小时内，在同一WiFi,mac2,os,ip2下交易的次数
    for item in ["wifi", "mac2", "os", "ip2"]:
        logging.info("day_hour_{}_count".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["trans_amt"].agg({"day_hour_{}_count".format(item):"count"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])
    
    #当前一小时内，在同一WiFi,mac2,os,ip2下交易的金额
    for item in ["wifi", "mac2", "os", "ip2"]:
        logging.info("day_hour_{}_transa_amt_sum".format(item))
        temp = transac.groupby(["new_day", "Hour", item], as_index=False)["trans_amt"].agg({"day_hour_{}_transa_amt_sum".format(item):"sum"})
        transac = pd.merge(transac, temp, how="left", on=["new_day", "Hour", item])

    #当天内，每个商户UID unique 购买次数
    logging.info("day_merchant_UID_unique")
    temp = transac.groupby(["new_day", "merchant"], as_index=False)["UID"].agg({"day_merchant_UID_unique":"nunique"})
    transac = pd.merge(transac, temp, on=["new_day", "merchant"], how="left")

    logging.info("day_merchant_UID_count")
    temp = transac.groupby(["new_day", "merchant"], as_index=False)["UID"].agg({"day_merchant_UID_count":"count"})
    transac = pd.merge(transac, temp, on=["new_day", "merchant"], how="left")
  
    return transac


def concat_data(train, test):
    data = pd.concat([train, test])
    return data

def split_tran_test(data):
    train = data.loc[data.Tag != -1]
    test = data.loc[data.Tag == -1]
    return train, test

def get_preFeature():
    logging.info("loading...")
    opera_train, transac_train = load_data("train")
    opera_test, transac_test = load_data("test")
    
    opera = concat_data(opera_train, opera_test)
    transac = concat_data(transac_train, transac_test)
    
    logging.info("opera...")
    opera = opera_feature(opera)
    
    logging.info("splite train test...")
    opera_train, opera_test = split_tran_test(opera)
    
    logging.info("writing the opera files...")
    opera_train.to_csv(config.NEW_RAW_FEATURE_OPERA_TRAIN, index=False)
    opera_test.to_csv(config.NEW_RAW_FEATURE_OPERA_TEST, index=False)
    
    logging.info("transac...")
    transac = transac_feature(transac)
    
    logging.info("split train test...")
    transac_train, transac_test = split_tran_test(transac)
    
    
    logging.info("writing the transac file...")
    transac_train.to_csv(config.NEW_RAW_FEATURE_TRANSAC_TRAIN, index=False)
    transac_test.to_csv(config.NEW_RAW_FEATURE_TRANSAC_TEST, index=False)
#    logging.info("opera start...")
#    opera_train = opera_feature(opera_train)
#    opera_test = opera_feature(opera_test)
#    
#    logging.info("writing the opera files...")
#    opera_train.to_csv(config.NEW_RAW_FEATURE_OPERA_TRAIN, index=False)
#    opera_test.to_csv(config.NEW_RAW_FEATURE_OPERA_TEST, index=False)
#    #===============================这是分割线==================================
#    
#    logging.info("transac start...")
#    transac_train = transac_feature(transac_train)
#    transac_test = transac_feature(transac_test)
#    
#    logging.info("writing the transac file...")
#    transac_train.to_csv(config.NEW_RAW_FEATURE_TRANSAC_TRAIN, index=False)
#    transac_test.to_csv(config.NEW_RAW_FEATURE_TRANSAC_TEST, index=False)
    
    logging.info("done")


if __name__ == "__main__":
    get_preFeature()


