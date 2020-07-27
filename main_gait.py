import pandas as pd
from util.utils import create_userids, print_list
from util.autoencoder import train_autoencoder

from util.augment_data import get_augmented_dataset
from util.oneclass import evaluate_authentication,  evaluate_authentication_cross_day
from util.normalization  import normalize_rows
from util.classification import evaluate_identification_CV
from util.plot import plot_raw_data

import util.settings as st
from util.model import train_model, evaluate_model, get_model_output_features
from util.autoencoder import get_autoencoder_output_features
from util.utils import create_userids,  create_userid_dictionary, create_bigram_100_csv
from util.classification import evaluate_identification_CV, evaluate_identification_Train_Test

from util.settings import AugmentationType


# creates a dataframe containing the data
# of a set of users [start_user, stop_user)
# 
def sub_dataframe(df, start_user, stop_user):
    users_list = generate_users_list(start_user, stop_user)
    new_df = df.loc[df.iloc[:, -1].isin(users_list)]
    print(new_df.shape)
    return new_df


# ------------------------------------------- VERIFICATION --------------------------------------------------------

# training: IDNet
# evaluation ZJU_Gait 
# 
def evaluate_autoencoder_authentication(training = False, augm = False, augm_type=AugmentationType.RND,   verbose = False, filename ='results/autoencoder_scores.csv'):
    if augm == True:
        if augm_type == AugmentationType.RND:
            model_name = "gait_autoencoder_fcn_rnd.h5"
        else:
            model_name = "gait_autoencoder_fcn_cshift.h5"
    else:
        model_name = "gait_autoencoder_fcn.h5"
    if training  == True:
        df_idnet = pd.read_csv("input_csv/idnet.csv", header = None) 
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)
        train_autoencoder(df_idnet, model_name=model_name, num_epochs=100)

    encoder_name = "encoder_"+model_name
    
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)


    features1 = get_autoencoder_output_features( df1, encoder_name )
    features2 = get_autoencoder_output_features( df2, encoder_name )

    auc1, eer1 = evaluate_authentication(features1, verbose)
    auc2, eer2 = evaluate_authentication(features2, verbose)
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, verbose )

    # dictionary of lists  
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(filename, index=False)


# training: IDNet
# evaluation ZJU_Gait 
# 
def evaluate_endtoend_authentication(training = False, augm = True, augm_type = AugmentationType.RND, verbose = False, filename ='results/endtoend_scores.csv'):
    if augm == True:
        if augm_type == AugmentationType.RND:
            model_name = "gait_fcn_rnd.h5"
        else:
            model_name = "gait_fcn_cshift.h5"
    else:
        model_name = "gait_fcn.h5"
    
    if training  == True:
        df_idnet = pd.read_csv("input_csv/idnet.csv", header = None)  
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)      
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        train_model(df_idnet, model_name)

    
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 = get_model_output_features( df1, model_name )
    features2 = get_model_output_features( df2, model_name )

    auc1, eer1 = evaluate_authentication(features1, verbose)
    auc2, eer2 = evaluate_authentication(features2, verbose)
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, verbose )

    # dictionary of lists  
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(filename, index=False)


def evaluate_raw_authentication(training = False, verbose = False, filename ='results/raw_scores.csv'):    
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)


    features1 =  df1
    features2 =  df2

    auc1, eer1 = evaluate_authentication(features1, verbose)
    auc2, eer2 = evaluate_authentication(features2, verbose)
    auc_cross, eer_cross = evaluate_authentication_cross_day( features1, features2, verbose )

    # dictionary of lists  
    dict = {'AUC1': auc1, 'EER1': eer1, 'AUC2': auc2, 'EER2': eer2, 'AUC_cross': auc_cross, 'EER_cross': eer_cross}    
    df = pd.DataFrame(dict) 
    df.to_csv(filename, index=False)

# ------------------------------------------- IDENTIFICATION --------------------------------------------------------

# training: IDNet
# evaluation ZJU_Gait 
# 
def evaluate_autoencoder_identification(training = False, augm = False, augm_type=AugmentationType.RND):
    if augm == True:
        if augm_type == AugmentationType.RND:
            model_name = "gait_autoencoder_fcn_rnd.h5"
        else:
            model_name = "gait_autoencoder_fcn_cshift.h5"
    else:
        model_name = "gait_autoencoder_fcn.h5"
    if training  == True:
        df_idnet = pd.read_csv("input_csv/idnet.csv", header = None) 
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)
        train_autoencoder(df_idnet, model_name=model_name, num_epochs=10)

        
        
    encoder_name = "encoder_"+model_name
    
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)


    features1 = get_autoencoder_output_features( df1, encoder_name )
    features2 = get_autoencoder_output_features( df2, encoder_name )

    evaluate_identification_CV(features1, num_folds=10)
    evaluate_identification_CV(features2, num_folds=10)
    evaluate_identification_Train_Test(features1, features2)


# training: IDNet
# evaluation ZJU_Gait
# 
def evaluate_endtoend_identification(training = False, augm = False, augm_type=AugmentationType.RND):
    if augm == True:
        if augm_type == AugmentationType.RND:
            model_name = "gait_fcn_rnd.h5"
        else:
            model_name = "gait_fcn_cshift.h5"
    else:
        model_name = "gait_fcn.h5"
    
    if training  == True:
        df_idnet = pd.read_csv("input_csv/idnet.csv", header = None)  
        if augm == True:       
            df_idnet = get_augmented_dataset(df_idnet, augm_type)      
        df_idnet = normalize_rows( df_idnet, st.NormalizationType.ZSCORE)
        train_model(df_idnet, model_name)
    
    
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    features1 = get_model_output_features( df1, model_name)
    features2 = get_model_output_features( df2, model_name)

    evaluate_identification_CV(features1, num_folds=10)
    evaluate_identification_CV(features2, num_folds=10)
    evaluate_identification_Train_Test(features1, features2)



def evaluate_raw_identification():
    df1 = pd.read_csv("input_csv/zju_session1_frames_raw.csv", header = None)
    df2 = pd.read_csv("input_csv/zju_session2_frames_raw.csv", header = None)
    
    df1 = normalize_rows( df1, st.NormalizationType.ZSCORE)
    df2 = normalize_rows( df2, st.NormalizationType.ZSCORE)

    evaluate_identification_CV(df1, num_folds=10)
    evaluate_identification_CV(df2, num_folds=10)
    evaluate_identification_Train_Test(df1, df2)

# ------------------------------------------- END IDENTIFICATION --------------------------------------------------------





def generate_features(augm , augm_type):
    if augm == True:
        if augm_type == AugmentationType.RND:
            model_name_endtoend = "gait_fcn_rnd.h5"
            model_name_autoencoder = 'encoder_gait_autoencoder_fcn_rnd.h5'
        else:
            model_name_endtoend = "gait_fcn_cshift.h5"
            model_name_autoencoder = 'encoder_gait_autoencoder_fcn_cshift.h5'
    else:
        model_name_endtoend = 'gait_fcn.h5'
        model_name_autoencoder = 'encoder_gait_autoencoder_fcn.h5'
    
    df = pd.read_csv("input_csv/zju_session0_frames_raw.csv", header = None)
    df = normalize_rows( df, st.NormalizationType.ZSCORE)
    # df.to_csv('output_csv/session0_raw.csv', index=False)

    features_endtoend    = get_model_output_features( df, model_name_endtoend)
    features_endtoend.to_csv('output_csv/session0_endtoend'+ '_'+ augm_type.value +'.csv', index=False)
    faetures_autoencoder = get_autoencoder_output_features( df, model_name_autoencoder)
    faetures_autoencoder.to_csv('output_csv/session0_autoencoder' + '_'+ augm_type.value +'.csv', index=False)

# ------------------------------------------- MAIN --------------------------------------------------------


# AUTHENTICATION

# evaluate_raw_authentication()
# evaluate_autoencoder_authentication(training = False, augm = False, augm_type = AugmentationType.CSHIFT, verbose = False)
# evaluate_endtoend_authentication(training = False, augm = False, augm_type = AugmentationType.RND, verbose = False)


# IDENTIFICATION

evaluate_raw_identification()
# evaluate_autoencoder_identification(training = False, augm = True, augm_type = AugmentationType.CSHIFT)
# evaluate_endtoend_identification(training = False)

# ------------------------------------------- END MAIN --------------------------------------------------------
