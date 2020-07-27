import os
import time
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras

from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn.model_selection import train_test_split
from sklearn import metrics

from util.utils import create_userids
from util.fcn import build_fcn
from util.resnet import build_resnet
from util.mlp import build_mlp
from util.tlenet import build_tlenet
import util.settings as stt

# ENS-TO-END models

def train_model( df, model_name = "foo.h5" ):
    userids = create_userids( df )
    nbclasses = len(userids)
    print(nbclasses)
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=stt.RANDOM_STATE)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=stt.RANDOM_STATE)

    print(X_train.shape)
    print(X_test.shape)
    print(X_val.shape)

    mini_batch_size = int(min(X_train.shape[0]/10, stt.BATCH_SIZE))
    if( model_name == "foo.h5"):
        model_name = stt.MODEL_NAME
    filepath = stt.TRAINED_MODELS_PATH + "/" + model_name

    if( stt.MODEL_TYPE == stt.ModelType.FCN ):
        cb, model = build_fcn((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    if( stt.MODEL_TYPE == stt.ModelType.RESNET ):
        cb, model = build_resnet((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    if( stt.MODEL_TYPE == stt.ModelType.MLP ):
        cb, model = build_mlp((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    if( stt.MODEL_TYPE == stt.ModelType.MCDCNN ):
        cb, model = build_mcdcnn((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    if( stt.MODEL_TYPE == stt.ModelType.TLENET ):
        cb, model = build_tlenet((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    if( stt.MODEL_TYPE == stt.ModelType.CNN ):
        cb, model = build_cnn((stt.FEATURES, stt.DIMENSIONS ), nbclasses, filepath )
    
    # if stt.UPDATE_WEIGHTS == True:
    #     model = set_weights_from_pretrained_model(model)

    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)

    # convert to tensorflow dataset
    train_ds = tf.data.Dataset.from_tensor_slices((X_train, y_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, y_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    start_time = time.time()
    hist = model.fit(train_ds, 
                      epochs=stt.EPOCHS,
                      verbose=True, 
                      validation_data=val_ds, 
                      callbacks=cb)
    
    hist_df = pd.DataFrame(hist.history) 

    # save history to csv: 
    hist_csv_file = 'histories/history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    duration = time.time() - start_time
    print("Training duration: "+str(duration/60))
    

    # EVALUATION 
    X_test = np.asarray(X_test).astype(np.float32)    
    y_true = np.argmax( y_test, axis=1)
    y_pred = np.argmax( model.predict(X_test), axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred)     
    print(accuracy)
    return model

# Evaluate model on a dataframe
# df: dataframe
def evaluate_model( df, model_name ):
    print("Evaluate model: ")
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]

    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    # evaluate model   
    model = tf.keras.models.load_model(stt.TRAINED_MODELS_PATH + "/" + model_name)
    # model.summary()

    y_true = np.argmax( y, axis=1)
    X = np.asarray(X).astype(np.float32)

    y_pred = np.argmax( model.predict(X), axis=1)
    accuracy = metrics.accuracy_score(y_true, y_pred)     
    print(accuracy)



# Use a pretrained model for feature extraction
# Load the model, pop the last layer
def get_model_output_features( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
  
    model = tf.keras.models.load_model(model_path)
    model.summary()
    print(model_name)
    model._layers.pop()
    model.outputs = [model.layers[-1].output]
    

    X = np.asarray(X).astype(np.float32)

    features = model.predict( X )
    df = pd.DataFrame( features )
    df['user'] = y 
    df.to_csv('features.csv', header = False, index=False)  
    return df




# def set_weights_from_pretrained_model(model):
#     model_path = stt.TRAINED_MODELS_PATH + '/' + stt.OLD_MODEL_NAME
#     try:
#         old_model = tf.keras.models.load_model(model_path)
#     except:
#         raise Exception(model_path + ' model does not exist!')
    
#     print('setting weights from '+model_path)

#     # Save the old model into SAVED_MODELS
#     if not os.path.exists(stt.SAVED_MODELS_PATH):
#         os.makedirs(stt.SAVED_MODELS_PATH)
#     old_model.save(stt.SAVED_MODELS_PATH + "/" + stt.MODEL_NAME)  

#     # Copy old model's weights to model
#     # The last layer weights will be ignored
#     for i in range(len(old_model.layers) - 1):
#         model.layers[i].set_weights(old_model.layers[i].get_weights())
#     return model
