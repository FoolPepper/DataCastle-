# -*- coding: utf-8 -*-
"""
Created on Fri Dec 14 22:15:38 2018

@author: FoolPepper
"""
from data_cleaning import data_Cleaning_run
from get_feature_1 import get_feature_1
from get_feature_2 import get_feature_2
from get_feature_3 import get_feature_3
from get_feature_4 import get_feature_4
from stacking import drop_feature_and_stacking

if __name__ == "__main__":
    print("start..")
    data_Cleaning_run()
    get_feature_1()
    get_feature_2()
    get_feature_3()
    get_feature_4()
    drop_feature_and_stacking()
    print("done!!!!!!!!!")


































