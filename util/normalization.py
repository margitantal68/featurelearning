import pandas as pd
import numpy as np
import util.settings as st

from sklearn import preprocessing
from sklearn.preprocessing import MinMaxScaler
from util.utils import create_userids

def normalize_rows( df, norm_type ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    rows, cols = X.shape
    if( norm_type == st.NormalizationType.MINMAX ):
        for i in range(0, rows):
            row = X[i,:]
            maxr = max(row)
            minr = min( row)
            if( maxr != minr ):
                X[i,:] = (X[i,:]- minr) /(maxr - minr)
            else:
                X[i,:] = 1
    if( norm_type == st.NormalizationType.ZSCORE ):
        for i in range(0, rows):
            row = X[i,:]
            mu = np.mean( row )
            sigma = np.std( row )
            if( sigma == 0 ):
                sigma = 0.0001
            X[i,:] = (X[i,:] - mu) / sigma
            
    df = pd.DataFrame( X )
    df['user'] = y 
    return df


def normalize_columns( df, norm_type ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures - 1
    X = array[:, 0:nfeatures]
    y = array[:, -1]
    
    if( norm_type == st.NormalizationType.MINMAX ):
        scaler = MinMaxScaler(feature_range=(0, 1))
        X = scaler.fit_transform(X)
    if( norm_type == st.NormalizationType.ZSCORE):
        X = preprocessing.scale( X )    
    df = pd.DataFrame( X )
    df['user'] = y 
    return df

def normalize_users_columns( df, norm_type):
    print(df.shape)
    userids = create_userids( df )
    user_data = df.loc[df.iloc[:, -1].isin([userids[0]])]    
    user_array= user_data.values
    nsamples, nfeatures = user_array.shape
    nfeatures = nfeatures - 1
    user_X = user_array[:, 0:nfeatures]
    user_y = user_array[:, -1]
    
    scaler = MinMaxScaler()
    print(userids[0]+": "+ str(user_X.shape))
    if( norm_type == st.NormalizationType.MINMAX ):
        user_X = scaler.fit_transform(user_X)
    if( norm_type == st.NormalizationType.ZSCORE):
        user_X = preprocessing.scale( user_X )    
    X = user_X
    y = user_y

    NUM_USERS = len(userids)
    for i in range(1,NUM_USERS):
        userid = userids[i]
        user_data = df.loc[df.iloc[:, -1].isin([userid])]
        user_array= user_data.values
        nsamples, nfeatures = user_array.shape
        nfeatures = nfeatures - 1
        user_X = user_array[:, 0:nfeatures]
        user_y = user_array[:, -1]
        
        if( norm_type == st.NormalizationType.MINMAX ):
            user_X = scaler.fit_transform(user_X)
        if( norm_type == st.NormalizationType.ZSCORE):
            user_X = preprocessing.scale( user_X )  
        # append data
        X = np.vstack([X, user_X])
        y = np.concatenate([y, user_y])
    df = pd.DataFrame( X )
    df['user'] = y 
    return df
