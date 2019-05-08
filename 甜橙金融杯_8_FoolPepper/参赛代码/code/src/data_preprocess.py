# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 21:36:10 2018

@author: FoolPepper
"""

import pandas as pd
import time
from datetime import datetime

import logging

import config

logging.basicConfig(
        level = logging.INFO,
        format = "[%(asctime)s] %(message)s",
        datefmt = "%Y-%m-%d %H:%M:%S",
        )

def load_data(data_type="train"):
    logging.info("loading {} data...".format(data_type))
    time_point = time.time()
    if data_type == "train":
        opera = pd.read_csv(config.RAW_DATA_OPERA_TRAIN, low_memory=False)
        transac = pd.read_csv(config.RAW_DATA_TRANSAC_TRAIN, low_memory=False)
        label = pd.read_csv(config.RAW_DATA_LABEL_TRAIN, low_memory=False)
        
#        opera = opera.fillna(-1)      #新增20181215
#        transac = transac.fillna(-1)
    else:
        opera = pd.read_csv(config.RAW_DATA_OPERA_TEST, low_memory=False)
        transac = pd.read_csv(config.RAW_DATA_TRANSAC_TEST, low_memory=False)
        label = pd.read_csv(config.RAW_DATA_LABEL_TEST, low_memory=False)
        label["Tag"] = -1
        
#        opera = opera.fillna(-1)
#        transac = transac.fillna(-1)
        
    logging.info("loading done, cost time {}s".format(time.time()-time_point))
    
    logging.info("merge Tag...")
    opera = pd.merge(opera, label, how="left", on=["UID"])
    transac = pd.merge(transac, label, how="left", on=["UID"])
    
    return opera, transac

def convert_device(row):
    row = str(row)
    if row == "nan":
        return "nan"
    if any(item in row for item in ["OPPO", "3005", "A51", "X9077", "R6", "PADT00", "PADM00", "1105", "R11", "R7", "R5", "R8", "X909", "R9", "A31U", "A31C"]):
        return "OPPO"
    elif any(item in row for item in ["IPHONE", "iPhone", "IPAD", "IPOD"]):
        return "Apple"
    elif any(item in row for item in ["HTC", "HONOR", "G620-L75", "G13", "G14", "G5"]):
        return "HTC"
    elif any(item in row for item in ["SONY", "D6633", "E6683", "G8142", "H4233"]):
        return "SONY"
    elif any(item in row for item in ["U8220", "C8812", "BAC", "V9", "PE-TL00M", "P6", "VTR", "PIC", "G621-TL00", "ALP", "HUAWEI", "ATH", "C8813", "CHM", "CUN", "PLE", "HWI", "NEM", "U8150", "SCL", "CHE", "ALE", "BTV", "HOL", "JDN", "GEM", "DUK", "LND", "KIW", "CLT", "ANE", "CPN", "V10", "FDR", "H60", "X1 7.0", "7S", "M200", "AGS", "H60", "DIG", "RNE", "FRD", "RNE", "COL", "MHA", "AL40", "CL00", "TL10", "TL20", "AL20", "AL10", "AL00", "LLD", "VKY", "KNT", "BND", "BKL", "TRT", "BLN", "CAM", "EML", "JMM", "BLA", "MYA", "PRA", "FIG", "ATU", "BLN"]):
        return "HUAWEI"
    elif any(item in row for item in ["MI", "SKR", "2014112", "2014501", "2014821", "2014011", "201481", "2013022", "XIAOMI", "红米", "HM"]):
        return "XiaoMi"
    elif any(item in row for item in ["魅", "M7", "M4", "Y685", "M3", "U20", "M2", "M6", "M3 NOTE", "M681C", "MX7", "MX5", "MX6", "MX4", "M1", "PRO", "MEIZU", "M6 NOTE", "M3 MAX", "M355", "M5 NOTE", "M5", "PRO 6", "PRO 7"]):
        return "MEIZU"
    elif any(item in row for item in ["PHILIPS"]):
        return "PHILIPS"
    elif any(item in row for item in ["N958ST", "ZTE", "K3D"]):
        return "ZTE"
    elif any(item in row for item in ["LENOVO", "ZUK"]):
        return "LENOVO"
    elif any(item in row for item in ["VIVO", "X7", "Y83", "V9S"]):
        return "VIVO"
    elif any(item in row for item in ["HLTE", "T928", "HISENSE", "HS"]):
        return "HISENSE"
    elif any(item in row for item in ["S9 PLUS", "NOTE5", "SCH", "SM", "SAMSUNG", "GT", "SGH-T959"]):
        return "SAMSUNG"
    elif any(item in row for item in ["CLIQ", "MOTO", "XT615"]):
        return "MOTO"
    elif any(item in row for item in ["MTS", "A01", "C106-9", "C105", "COOLPAD", "C106-7"]):
        return "COOLPAD"
    elif any(item in row for item in ["G011", "G0121", "G0"]):
        return "GREE"
    elif any(item in row for item in ["LG", "VS"]):
        return "LG"
    elif any(item in row for item in ["NOKIA", "TA"]):
        return "NOKIA"
    elif any(item in row for item in ["TCL", "P318L", "P1 S"]):
        return "TCL"
    elif any(item in row for item in ["LEPHONE", "X608", "X900", "LEX", "LE X621", "LE X", "X600", "LETV", "LE X620"]):
        return "LETV"
    elif any(item in row for item in ["T1", "T3", "T2", "OD103", "OS105", "YQ", "OC"]):
        return "T"
    elif any(item in row for item in ["F100S", "F100SL", "GN", "W800", "W900", "F100", "S9L"]):
        return "JINLI"
    elif any(item in row for item in ["ONEPLUS", "ONE", "A1001"]):
        return "1+"
    elif any(item in row for item in ["NEXUS"]):
        return "NEXUS"
    elif any(item in row for item in ["T8", "MP"]):
        return "MEITU"
    elif any(item in row for item in ["NX"]):
        return "NUBIA"
    elif any(item in row for item in ["HT-", "HAIER"]):
        return "HAIER"
    elif any(item in row for item in ["ASUS"]):
        return "ASUS"
    elif any(item in row for item in ["C1530L", "C1230L", "PHICOMM", "C1330", "X800"]):
        return "PHICOMM"
    elif any(item in row for item in ["VIRTUAL MACHINE"]):
        return "VIRTUAL MACHINE"
    else:
        return "OTHERS"

def convert_datetime_hour(row):
    opera_time = datetime.strptime(row, "%H:%M:%S")
    return opera_time.hour

def convert_datetime_minute(row):
    opera_time = datetime.strptime(row, "%H:%M:%S")
    return opera_time.minute

def convert_datetime_second(row):
    opera_time = datetime.strptime(row, "%H:%M:%S")
    return opera_time.second


def geo_decode(row):
    """
    Notes
    ------
    geo code暂时不考虑将其转化为经纬坐标
    """
    return 0


def opera_version(row):
    """
    Notes
    -----
    版本号只考虑前三位
    """
    if not isinstance(row, str):
        return -1
    row = row[0:5].replace(".", "")
    return int(row)


def deal(opera, transac):
    """
    Params
    -----
    opera：DataFrame
    transac：DataFrame
    
    Returns
    -----
    opera：DataFrame
    transac：DataFrame
    
    Notes
    -----
    数据处理：转化时间信息， 改变手机品牌，填充空白信息。
    """
    logging.info("convert device...")
    opera["device_type"] = opera["device2"].apply(convert_device)
    transac["device_type"] = transac["device2"].apply(convert_device)
    
    logging.info("Opera hour...")
    opera["Hour"] = opera["time"].apply(convert_datetime_hour)
    
    logging.info("Opera minute...")
    opera["Minute"] = opera["time"].apply(convert_datetime_minute)
    
    logging.info("Opera second...")
    opera["Second"] = opera["time"].apply(convert_datetime_second)
    
    logging.info("Transac hour...")
    transac["Hour"] = transac["time"].apply(convert_datetime_hour)
    
    logging.info("Transac minute...")
    transac["Minute"] = transac["time"].apply(convert_datetime_minute)
    
    logging.info("Transac second...")
    transac["Second"] = transac["time"].apply(convert_datetime_second)
    
    logging.info("Transac 3 hour...")
    transac["three_hour"] = transac.Hour.apply(lambda x: x//3)

    logging.info("Transac 6 hour...")
    transac["six_hour"] = transac.Hour.apply(lambda x: x//6)
    
    logging.info("Transac 12 hour...")
    transac["twelve_hour"] = transac.Hour.apply(lambda x: x//12)

    logging.info("version...")
    opera["version_num"] = opera.version.apply(opera_version)
        
    logging.info("fill nan...")
    opera = opera.fillna(-1)
    transac = transac.fillna(-1)
    
    return opera, transac   


def cross_feature(opera, transac):
    
    opera_ = opera[["UID", "day", "time", "wifi", "mac2", "mode", "os", "ip2"]]
    transac = pd.merge(transac, opera_, how="left", on=["UID", "day", "time"])
    return transac

def set_weekend(data_type, opera, transac):
    if data_type == "train":
        logging.info("weekend...")
        opera["weekend"] = 0
        opera.loc[opera.day.isin([1,2,8,9,15,16,22,23,29,30])  ,"weekend"] = 1
        transac["weekend"] = 0
        transac.loc[transac.day.isin([1,2,8,9,15,16,22,23,29,30])  ,"weekend"] = 1
    else:
        logging.info("weekend...")
        opera["weekend"] = 0
        opera.loc[opera.day.isin([6,7,13,14,20,21,27,28])  ,"weekend"] = 1
        transac["weekend"] = 0
        transac.loc[transac.day.isin([6,7,13,14,20,21,27,28])  ,"weekend"] = 1
    return opera, transac

def new_day(opera, transac):
    opera["new_day"] = opera["day"] + 30
    transac["new_day"] = transac["day"] + 30
    return opera, transac

def fix(opera):
    opera["ip2"] = opera.ip1
    opera["ip1_sub"] = opera.ip1
    opera["ip2_sub"] = opera.ip1
    
    return opera

def fixx(transac):
    transac["code1"] = transac.ip1
    transac["code2"] = transac.ip1
    return transac

def data_preprocess():
    logging.info("loading...")
    opera_train, transac_train = load_data("train")
    opera_test, transac_test = load_data("test")
    
    opera_train = fix(opera_train)
    opera_test = fix(opera_test)
    
    transac_train = fixx(transac_train)
    transac_test = fixx(transac_test)
    
    
    logging.info("cross feature...")
    transac_train = cross_feature(opera_train, transac_train)
    transac_test = cross_feature(opera_test, transac_test)
    
    logging.info("deal...")
    opera_train, transac_train = deal(opera_train, transac_train)
    opera_test, transac_test = deal(opera_test, transac_test)
    
    
    
    logging.info("set weekend...")
    opera_train, transac_train = set_weekend("train", opera_train, transac_train)
    opera_test, transac_test = set_weekend("test", opera_test, transac_test)
    
    logging.info("new_day...")
    opera_train["new_day"] = opera_train.day
    transac_train["new_day"] = transac_train.day
    opera_test, transac_test = new_day(opera_test, transac_test)

    logging.info("writing to csv files...")
    opera_train.to_csv(config.PRE_DATA_OPERA_TRAIN, index=False)
    opera_test.to_csv(config.PRE_DATA_OPERA_TEST, index=False)
    transac_train.to_csv(config.PRE_DATA_TRANSAC_TRAIN, index=False)
    transac_test.to_csv(config.PRE_DATA_TRANSAC_TEST, index=False)
    logging.info("done")
    

if __name__ == "__main__":
    data_preprocess()          #ss
        