import pandas as pd
import numpy as np
import random
from util.settings import AugmentationType

random.seed(42)



def get_augmented_dataset(df, augm_type):
    if augm_type == AugmentationType.RND:
        return get_augmented_dataset_RND( df )
    if augm_type == AugmentationType.CSHIFT:
        return get_augmented_dataset_CSHIFT( df )


# input: the datframe to be augmented
# output: the augmented dataframe
# doubles the rows in the dataset  
def get_augmented_dataset_RND(df):
    aug_df = df.copy()    
    print(aug_df.shape)
    # select all columns but the last one
    aug_df_nouserid = aug_df.iloc[:,:-1]
    # Applying an augmentation function to all columns data
    aug_df_nouserid.loc[:] += np.random.uniform(-0.2, 0.2, size=aug_df_nouserid.shape)
    last_column = aug_df[aug_df.columns[-1]]
    df2 =  pd.concat([aug_df_nouserid, last_column], axis=1)
    return pd.concat([df, df2], axis=0)


# input: array, idx
# output: rotated array
# 

def rotate_array( iarray, len, idx ):
    temp = np.zeros( len )
    for j in range(idx, len):
        temp[j-idx] = iarray[ j ]
    for j in range(0, idx):
        temp[len-idx+j] = iarray[ j ]    
    return temp

# input: the datframe to be augmented
# output: the augmented dataframe
# doubles the rows in the dataset  
# 
def get_augmented_dataset_CSHIFT(df):
    aug_df = df.copy()    
    array = aug_df.values
    rows, cols = df.shape
    print(aug_df.shape)
    
    len = 128
    for i in range(0, rows):
        idx = ((int)(np.random.uniform(0, 300)) ) % len
        xarray = rotate_array( array[i, 0:len], len, idx)
        yarray = rotate_array( array[i, len: 2*len], len, idx)
        zarray = rotate_array( array[i, 2*len: 3*len], len, idx)
        for j in range(0, len):
            array[i, j] = xarray[j]
            array[i, j+len] = yarray[j]
            array[i, j+2*len] = zarray[j]
    return pd.concat([df, aug_df], axis=0)




# array = np.array([[1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 'u1'], [1, 2, 3, 4,  5, 6, 7, 8, 9, 10, 11, 12, 'u2']])
# df = pd.DataFrame(array)
# print(df)
# aug_df = get_augmented_dataset2( df )
# print(aug_df)


# # df = pd.read_csv("input_csv/zju_session0_frames_raw.csv", header = None)
# # aug_df = get_augmented_dataset(df)
# # print(aug_df.head)