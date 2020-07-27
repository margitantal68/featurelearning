import numpy as np
import util.normalization
from util.utils import create_userids
from util.plot import plot_ROC

from sklearn.svm import OneClassSVM
from sklearn import metrics
from util.settings import AGGREGATE_BLOCK_NUM



def calculate_EER(y, scores):
    # Calculating EER
    fpr,tpr,threshold = metrics.roc_curve(y,scores,pos_label=1)
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    return EER 

def compute_AUC(positive_scores, negative_scores):
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, thresholds = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    return roc_auc

def compute_AUC_EER(positive_scores, negative_scores):  
    zeros = np.zeros(len(negative_scores))
    ones  = np.ones(len(positive_scores))
    y = np.concatenate((zeros, ones))
    scores = np.concatenate((negative_scores, positive_scores))
    fpr, tpr, threshold = metrics.roc_curve(y, scores, pos_label=1)
    roc_auc = metrics.auc(fpr, tpr)
    # Calculating EER
    fnr = 1-tpr
    EER_threshold = threshold[np.argmin(abs(fnr-fpr))]
    
    # print EER_threshold
    EER_fpr = fpr[np.argmin(np.absolute((fnr-fpr)))]
    EER_fnr = fnr[np.argmin(np.absolute((fnr-fpr)))]
    EER = 0.5 * (EER_fpr+EER_fnr)
    
    return roc_auc, EER

def evaluate_authentication( df, verbose = False):
    print(df.shape)
    userids = create_userids( df )
    NUM_USERS = len(userids)
    auc_list = list()
    eer_list = list()
    global_positive_scores = list()
    global_negative_scores = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]
        user_train_data = df.loc[ df.iloc[:, -1].isin([userid]) ]
        # Select data for training
        user_train_data = user_train_data.drop(user_train_data.columns[-1], axis=1)
        user_array = user_train_data.values
 
        num_samples = user_array.shape[0]
        train_samples = (int)(num_samples * 0.66)
        test_samples = num_samples - train_samples
        # print("#train_samples: "+str(train_samples)+"\t#test_samples: "+ str(test_samples))
        user_train = user_array[0:train_samples,:]
        user_test = user_array[train_samples:num_samples,:]
     
        other_users_data = df.loc[~df.iloc[:, -1].isin([userid])]
        other_users_data = other_users_data.drop(other_users_data.columns[-1], axis=1)
        other_users_array = other_users_data.values   
        
        clf = OneClassSVM(gamma='scale')
        clf.fit(user_train)
 
        positive_scores = clf.score_samples(user_test)
        negative_scores =  clf.score_samples(other_users_array)   
        
        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        auc, eer = compute_AUC_EER(y_pred_positive, y_pred_negative)
        # auc, eer = compute_AUC_EER(positive_scores, negative_scores )

        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        if  verbose == True:
            print(str(userid)+", "+ str(auc)+", "+str(eer) )
         
        auc_list.append(auc)
        eer_list.append(eer)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )
    
    if verbose == True:
        global_auc, global_eer = compute_AUC_EER(global_positive_scores, global_negative_scores)
        print("Global AUC: "+str(global_auc))
        print("Global EER: "+str(global_eer))
    return auc_list, eer_list

# Used only for Gait authentication
# df1 - ZJU_Gait_session1
# df2 - ZJU_Gait_session2

def evaluate_authentication_cross_day( df1, df2, verbose = False ):
    print("Session 1 shape: "+str(df1.shape))
    print("Session 2 shape: "+str(df2.shape))
        
    userids = create_userids( df1 )
    NUM_USERS = len(userids)
    
    global_positive_scores = list()
    global_negative_scores = list()
    auc_list = list()
    eer_list = list()
    for i in range(0,NUM_USERS):
        userid = userids[i]

        user_session1_data = df1.loc[df1.iloc[:, -1].isin([userid])]
        user_session2_data = df2.loc[df2.iloc[:, -1].isin([userid])]
      
        user_session1_data = user_session1_data.drop(user_session1_data.columns[-1], axis=1)
        user_session1_array = user_session1_data.values
 
        # positive test data
        user_session2_data =  user_session2_data.drop(user_session2_data.columns[-1], axis=1) 
        user_session2_array = user_session2_data.values

        # negative test data
        other_users_session2_data = df2.loc[~df2.iloc[:, -1].isin([userid])]
        other_users_session2_data = other_users_session2_data.drop(other_users_session2_data.columns[-1], axis=1)
        other_users_session2_array = other_users_session2_data.values   
        
        clf = OneClassSVM(gamma='scale')
        clf.fit(user_session1_array)
 
        positive_scores = clf.score_samples(user_session2_array)
        negative_scores =  clf.score_samples(other_users_session2_array)   

        # Aggregating positive scores
        y_pred_positive = positive_scores
        for i in range(len(positive_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_positive[i] = np.average(y_pred_positive[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        # Aggregating negative scores
        y_pred_negative = negative_scores
        for i in range(len(negative_scores) - AGGREGATE_BLOCK_NUM + 1):
            y_pred_negative[i] = np.average(y_pred_negative[i : i + AGGREGATE_BLOCK_NUM], axis=0)

        auc, eer = compute_AUC_EER(y_pred_positive, y_pred_negative)

        
        # auc, eer = compute_AUC_EER(positive_scores, negative_scores )
 
        global_positive_scores.extend(positive_scores)
        global_negative_scores.extend(negative_scores)

        if verbose == True:
            print(str(userid)+": "+ str(auc)+", "+str(eer) )
        auc_list.append(auc)
        eer_list.append(eer)
    print('AUC  mean : %7.4f, std: %7.4f' % ( np.mean(auc_list), np.std(auc_list)) )
    print('EER  mean:  %7.4f, std: %7.4f' % ( np.mean(eer_list), np.std(eer_list)) )

    if verbose == True:
        global_auc, global_eer = compute_AUC_EER(global_positive_scores, global_negative_scores)
        print("Global AUC: "+str(global_auc))
        print("Global EER: "+str(global_eer))
    return auc_list, eer_list










