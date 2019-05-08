# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

import lightgbm as lgb
import xgboost as xgb
#from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd 
from sklearn.model_selection import StratifiedKFold



from drop_kde_fea import  drop_col_KDE_ss1Clean, drop_col_KDE_xx
#from col_nor import col_normalize, ss_feature_list_3, bb_feature_list_3
from drop_day_trans_amt_bal import get_drop_day_trans_amt_bal

def tpr_weight_funtion(y_true,y_predict):
    d = pd.DataFrame()
    d['prob'] = list(y_predict)
    d['y'] = list(y_true)
    d = d.sort_values(['prob'], ascending=[0])
    y = d.y
    PosAll = pd.Series(y).value_counts()[1]
    NegAll = pd.Series(y).value_counts()[0]
    pCumsum = d['y'].cumsum()
    nCumsum = np.arange(len(y)) - pCumsum + 1
    pCumsumPer = pCumsum / PosAll
    nCumsumPer = nCumsum / NegAll
    TR1 = pCumsumPer[abs(nCumsumPer-0.001).idxmin()]
    TR2 = pCumsumPer[abs(nCumsumPer-0.005).idxmin()]
    TR3 = pCumsumPer[abs(nCumsumPer-0.01).idxmin()]
    return 'TC_AUC',0.4 * TR1 + 0.3 * TR2 + 0.3 * TR3,True



#op_train = pd.read_csv('../round2/operation_train_new.csv')
#trans_train = pd.read_csv('../round2/transaction_train_new.csv')
#
#op_test = pd.read_csv('../round2/test_operation_round2.csv')
#trans_test = pd.read_csv('../round2/test_transaction_round2.csv')
#y = pd.read_csv('../round2/tag_train_new.csv')
#sub = pd.read_csv('../round2/submit_example.csv')

def drop_feature_and_stacking():
    get_drop_day_trans_amt_bal()                  #drop
    
    ip_train = pd.read_csv('../Feature/ip1_feature_train.csv')
    ip_test = pd.read_csv('../Feature/ip1_feature_test.csv')
    
    #device2_train = pd.read_csv('../round2_fea/TransfromFeature_device2_train.csv')
    #device2_test = pd.read_csv('../round2_fea/TransfromFeature_device2_test.csv')
    
    #cc_fea_train = pd.read_csv('../round2_fea/cc_feature_train.csv')
    #cc_fea_test = pd.read_csv('../round2_fea/cc_feature_test.csv')
    
    #ss1_fea_train = pd.read_csv('../Feature/ss1_train.csv')
    #ss1_fea_test = pd.read_csv('../Feature/ss1_test.csv')
    
    bb_train = pd.read_csv('../Feature/bb_fea_train_clean_beDrop.csv')
    bb_test = pd.read_csv('../Feature/bb_fea_test_clean_beDrop.csv')
    
    ss1Clean_train = pd.read_csv('../Feature/ss1_train_clean.csv')
    ss1Clean_test = pd.read_csv('../Feature/ss1_test_clean.csv')
    
    xx_train = pd.read_csv('../Feature/same_time_and_muti_UID_train.csv')
    xx_test = pd.read_csv('../Feature/same_time_and_muti_UID_test.csv')
    
    #lw_train = pd.read_csv('../round2_clean/linwei_feature_train.csv')
    #lw_test = pd.read_csv('../round2_clean/linwei_feature_test.csv')
    
    #ss3_train = pd.read_csv('../round2_fea/ss3_train.csv')
    #ss3_test = pd.read_csv('../round2_fea/ss3_test.csv')
    #
    #ss4_train = pd.read_csv('../round2_fea/ss4_train.csv')
    #ss4_test = pd.read_csv('../round2_fea/ss4_test.csv')
    #
    #ss6_train = pd.read_csv('../round2_fea/ss6_train.csv')
    #ss6_test = pd.read_csv('../round2_fea/ss6_test.csv')
    #
    #lyf_train = pd.read_csv('../round2_fea/lyf_train.csv')
    #lyf_test = pd.read_csv('../round2_fea/lyf_test.csv')
    
    
    
    train = pd.merge(bb_train,ip_train,on='UID',how='left')
    #train = pd.merge(train,cc_fea_train,on='UID',how='left')
    train = pd.merge(train,ss1Clean_train,on='UID',how='left')
    train = pd.merge(train,xx_train,on='UID',how='right')
    #train = pd.merge(train,lw_train,on='UID',how='left')
    #train = pd.merge(train,ss3_train,on='UID',how='left')
    #train = pd.merge(train,ss4_train,on='UID',how='left')
    #train = pd.merge(train,ss6_train,on='UID',how='left')
    
    test = pd.merge(bb_test,ip_test,on='UID',how='left')
    #test = pd.merge(test,cc_fea_test,on='UID',how='left')
    test = pd.merge(test,ss1Clean_test,on='UID',how='left')
    test = pd.merge(test,xx_test,on='UID',how='left')
    #test = pd.merge(test,lw_test,on='UID',how='left')
    #test = pd.merge(test,ss3_test,on='UID',how='left')
    #test = pd.merge(test,ss4_test,on='UID',how='left')
    #test = pd.merge(test,ss6_test,on='UID',how='left')
    
    label = train['Tag']
    train = train.drop(['Tag'],axis = 1).fillna(-1)
    #label = y['Tag']
    
    #train = drop_col_KDE_bb(train)
    train = drop_col_KDE_ss1Clean(train)
    train = drop_col_KDE_xx(train)
    #train = drop_col_KDE_lyf(train)
    #train = drop_col_KDE_ss3(train)
    #train = drop_col_KDE_ss4(train)
    #train = drop_col_KDE_ss6(train)
    #train = drop_col_KDE_lw(train)
    
    #train = drop_day(train)
    
    #train = col_normalize(train, ss_feature_list_3)
    #train = col_normalize(train, bb_feature_list_3)
    
    test_id = test['UID']
    test = test.drop(['Tag'],axis = 1).fillna(-1)
    
    #test = drop_col_KDE_bb(test)
    test = drop_col_KDE_ss1Clean(test)
    test = drop_col_KDE_xx(test)
    #test = drop_col_KDE_lyf(test)
    #test = drop_col_KDE_ss3(test)
    #test = drop_col_KDE_ss4(test)
    #test = drop_col_KDE_ss6(test)
    #test = drop_col_KDE_lw(test)
    
    #test = drop_day(test)
    
    #test = col_normalize(test, ss_feature_list_3)
    #test = col_normalize(test, bb_feature_list_3)
    
    lgb_model = lgb.LGBMClassifier(boosting_type='gbdt', num_leaves=100, reg_alpha=3, reg_lambda=5, max_depth=-1,
        n_estimators=5000, objective='binary', subsample=0.9, colsample_bytree=0.77, subsample_freq=1, learning_rate=0.05,
        random_state=1000, n_jobs=16, min_child_weight=4, min_child_samples=5, min_split_gain=0)
    
    
    xgb_model = xgb.XGBClassifier(n_estimators=3000,min_child_weight=4,scale_pos_weight=6,subsample=0.9,colsample_bytree=0.8,
                                  learning_rate=0.1, objective = 'binary:logistic', eval_metric='auc',alpha = 4e-5,n_jobs=8)
    
    rf_model = RandomForestClassifier(n_estimators=3000, max_depth=8, max_features='sqrt', oob_score=True)
    
    gbdt_model = GradientBoostingClassifier(n_estimators=500, subsample=0.8, max_features=0.7)
    
    skf = StratifiedKFold(n_splits=5, random_state=2018, shuffle=True)
    best_score = []
    
    xgb_oof_preds = np.zeros(train.shape[0])
    lgb_oof_preds = np.zeros(train.shape[0])
    gbdt_oof_preds = np.zeros(train.shape[0])
    rf_oof_preds = np.zeros(train.shape[0])
    
    
    lgb_sub_preds = np.zeros(test_id.shape[0])
    xgb_sub_preds = np.zeros(test_id.shape[0])
    gbdt_sub_preds = np.zeros(test_id.shape[0])
    rf_sub_preds = np.zeros(test_id.shape[0])
    
    for index, (train_index, test_index) in enumerate(skf.split(train, label)):
        lgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose=50,
                      eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                                (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds=30)
        
        print('start xgb...')
        xgb_model.fit(train.iloc[train_index], label.iloc[train_index], verbose = 50, 
                      eval_set=[(train.iloc[train_index], label.iloc[train_index]),
                                (train.iloc[test_index], label.iloc[test_index])], early_stopping_rounds = 40,
                     eval_metric = 'auc')
        
        print('start rf...')
        rf_model.fit(train.iloc[train_index], label.iloc[train_index])
        
        print('start gbdt...')
        gbdt_model.fit(train.iloc[train_index], label.iloc[train_index])
        
        best_score.append(lgb_model.best_score_['valid_1']['binary_logloss'])
        print('the best_score of lightgbm is')
        print(best_score)
        
        lgb_oof_preds[test_index] = lgb_model.predict_proba(train.iloc[test_index], num_iteration=lgb_model.best_iteration_)[:,1]
        xgb_oof_preds[test_index] = xgb_model.predict_proba(train.iloc[test_index])[:,1]
        gbdt_oof_preds[test_index] = gbdt_model.predict_proba(train.iloc[test_index])[:,1]
        rf_oof_preds[test_index] = rf_model.predict_proba(train.iloc[test_index])[:,1]
    
        
        lgb_sub_preds += lgb_model.predict_proba(test, num_iteration=lgb_model.best_iteration_)[:, 1] / 5
        xgb_sub_preds += xgb_model.predict_proba(test)[:, 1] / 5
        rf_sub_preds += rf_model.predict_proba(test)[:,1] / 5
        gbdt_sub_preds += gbdt_model.predict_proba(test)[:,1] / 5
        
        
    
    new_train = np.concatenate([lgb_oof_preds.reshape(len(lgb_oof_preds),1),
                                xgb_oof_preds.reshape(len(xgb_oof_preds),1),
                                gbdt_oof_preds.reshape(len(gbdt_oof_preds),1),
                                rf_oof_preds.reshape(len(rf_oof_preds),1)], axis=1)
    new_test = np.concatenate([lgb_sub_preds.reshape(len(test_id),1),
                               xgb_sub_preds.reshape(len(test_id),1),
                               gbdt_sub_preds.reshape(len(test_id),1),
                               rf_sub_preds.reshape(len(test_id),1)], axis=1)
    
    from sklearn.linear_model import LogisticRegressionCV
    LR_model = LogisticRegressionCV(penalty='l2',solver = 'lbfgs',cv=5)
    LR_model.fit(new_train, label)
    sub_preds = LR_model.predict_proba(new_test)[:,1]
    
    
    #m = tpr_weight_funtion(y_predict=oof_preds,y_true=label)
    #print(m[1])
    sub = pd.read_csv('../RawData/submit_example.csv')
    sub['Tag'] = sub_preds
    sub.to_csv('../sub/result_%s.csv'%str(12_14_0),index=False)





