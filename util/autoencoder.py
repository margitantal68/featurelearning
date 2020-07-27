# 1. Set `PYTHONHASHSEED` environment variable at a fixed value
import os

import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import util.settings as stt

import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import backend as K

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder 
from sklearn import preprocessing


def build_FCN_autoencoder( input_shape, num_epochs, file_path):
    input_dim = stt.DIMENSIONS
    input_layer = keras.layers.Input(shape=input_shape)
    conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    print(K.int_shape(conv3))
    encoded = keras.layers.GlobalAveragePooling1D()(conv3)
    # print(K.int_shape(encoded))
    dim_encoded = K.int_shape(encoded)[1]
    # shape = K.int_shape(encoded)
    # print("shape of encoded {}".format(K.int_shape(encoded)))

    # DECODER
    # stt.FEATURES must be multiple of 128
    factor = 1
    if( dim_encoded < stt.FEATURES ):
        factor = (int) (stt.FEATURES / dim_encoded)
    print(factor)
  
    h = keras.layers.Reshape((dim_encoded, 1) )(encoded)
    h = keras.layers.UpSampling1D( factor )(h)  
    conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(h)
    conv3 = keras.layers.BatchNormalization()(conv3)
    conv3 = keras.layers.Activation('relu')(conv3)
    conv2 = keras.layers.Conv1D( filters=256, kernel_size=5, padding='same')(conv3)
    conv2 = keras.layers.BatchNormalization()(conv2)
    conv2 = keras.layers.Activation('relu')(conv2)
    conv1 = keras.layers.Conv1D(filters=input_dim, kernel_size=8, padding='same')(conv2)
    conv1 = keras.layers.BatchNormalization()(conv1)
    conv1 = keras.layers.Activation(activation='relu')(conv1)
    decoded = conv1

    encoder = keras.Model(input_layer, encoded)
    autoencoder = keras.Model(input_layer, decoded)
    autoencoder.compile(optimizer='adam', loss='mse')    
    reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
                                                  min_lr=0.0001) 
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose=1)
    callbacks = [reduce_lr,model_checkpoint]
    print(autoencoder.summary()) 
    return callbacks, encoder, autoencoder


# TRAIN and SAVE autoencoder
# Parameters:
#   df - dataframe 
#   model_name - e.g. 'autoencoder_FCN.h5'
#   num_epoch - number of epoch

def train_autoencoder(df, model_name, num_epochs=10):
    filepath = stt.TRAINED_MODELS_PATH + "/" + model_name
    # build autoencoder
    input_shape = (stt.FEATURES, stt.DIMENSIONS)
    file_path = stt.TRAINED_MODELS_PATH + "/" + model_name
    callbacks, encoder, model = build_FCN_autoencoder( input_shape, num_epochs, file_path)

    # split dataframe
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    
    enc = OneHotEncoder()
    enc.fit(y.reshape(-1,1))
    y = enc.transform(y.reshape(-1, 1)).toarray()
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)
 
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=stt.RANDOM_STATE)
    mini_batch_size = int(min(X_train.shape[0]/10, stt.BATCH_SIZE))

    # convert to tensorflow dataset
    X_train = np.asarray(X_train).astype(np.float32)
    X_val = np.asarray(X_val).astype(np.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((X_train, X_train))
    val_ds = tf.data.Dataset.from_tensor_slices((X_val, X_val))

    BATCH_SIZE = mini_batch_size
    SHUFFLE_BUFFER_SIZE = 100

    train_ds = train_ds.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
    val_ds = val_ds.batch(BATCH_SIZE)

    # train model
    history = model.fit(train_ds,
                              epochs=num_epochs,
                              shuffle=False,
                              validation_data=val_ds)
    save_and_plot_history( history, model_name )
    # Save the autoencoder and separately its encoder part (Used as a feature extractor!!!)
    print('Saved model: ' + model_name)
    model.save(stt.TRAINED_MODELS_PATH + '/' + model_name)
    encoder_name = 'encoder_' + model_name
    print('Saved encoder: ' + encoder_name)
    encoder.save(stt.TRAINED_MODELS_PATH + '/' + encoder_name)


def save_and_plot_history( history, model_name ):
    model_name_without_ext = model_name[0 : model_name.find('.')]
    print(model_name_without_ext)
    # convert to dataframe
    hist_df = pd.DataFrame(history.history) 
    # save to csv: 
    hist_csv_file = 'histories/'+model_name_without_ext + '_history.csv'
    with open(hist_csv_file, mode='w') as f:
        hist_df.to_csv(f)
    # Plot training & validation loss values
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train', 'Test'], loc='upper left')
    plt.show()



# Use a pretrained model for feature extraction
# Load the encoder
def get_autoencoder_output_features( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
   
    model = tf.keras.models.load_model(model_path)
    model.summary()
    X = np.asarray(X).astype(np.float32)

    features = model.predict( X )
    print(model_path)
    print(features.shape)
    df = pd.DataFrame( features )
    df['user'] = y 
    return df

# Use a pretrained model for samples generation
# df - input dataset
# model_name -  autoencoder
# return the generated data
def generate_autoencoder_samples( df, model_name ):
    array = df.values
    nsamples, nfeatures = array.shape
    nfeatures = nfeatures -1 
    X = array[:,0:nfeatures]
    y = array[:,-1]
    X = X.reshape(-1, stt.FEATURES, stt.DIMENSIONS)

    model_path = stt.TRAINED_MODELS_PATH + '/' + model_name
   
    model = tf.keras.models.load_model(model_path)
    model.summary()
    X = np.asarray(X).astype(np.float32)

    generated_samples = model.predict( X )
    print(model_path)
    print(generated_samples.shape)

    generated_samples = generated_samples.reshape(-1, stt.FEATURES * stt.DIMENSIONS)
    df = pd.DataFrame( generated_samples )
    df['user'] = y 
    return df



# def build_FCN_autoencoder( input_shape, num_epochs, file_path):
#     input_dim = stt.DIMENSIONS
#     input_layer = keras.layers.Input(shape=input_shape)
#     conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.Activation(activation='relu')(conv1)
#     conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.Activation('relu')(conv2)
#     conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.Activation('relu')(conv3)
#     print(K.int_shape(conv3))
#     encoded = keras.layers.GlobalAveragePooling1D()(conv3)
#     print(K.int_shape(encoded))
#     dim2 = K.int_shape(encoded)[1]
#     shape = K.int_shape(encoded)
#     print("shape of encoded {}".format(K.int_shape(encoded)))

#     # DECODER
#     h = keras.layers.Reshape((dim2, 1) )(encoded)
#     # Signature 128 --> 4 x 128 = 512
#     # h = keras.layers.UpSampling1D(4)( h )



#     conv3 = keras.layers.Conv1D(256, kernel_size=3, padding='same')(h)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.Activation('relu')(conv3)
#     conv2 = keras.layers.Conv1D( filters=128, kernel_size=5, padding='same')(conv3)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.Activation('relu')(conv2)
#     conv1 = keras.layers.Conv1D(filters=input_dim, kernel_size=8, padding='same')(conv2)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.Activation(activation='relu')(conv1)
#     decoded = conv1
#     print("shape of decoded {}".format(K.int_shape(decoded)))

#     encoder = keras.Model(input_layer, encoded)
#     autoencoder = keras.Model(input_layer, decoded)
#     autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])    
#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
#                                                   min_lr=0.0001) 
#     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose=1)
#     callbacks = [reduce_lr,model_checkpoint]
#     print(autoencoder.summary()) 
#     return callbacks, encoder, autoencoder

# # POOR !!!
# def build_FCN_autoencoder( input_shape, num_epochs, file_path):
#     input_dim = stt.DIMENSIONS
#     input_layer = keras.layers.Input(shape=input_shape)
#     conv1 = keras.layers.Conv1D(filters=128, kernel_size=8, padding='same')(input_layer)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.Activation(activation='relu')(conv1)
#     conv2 = keras.layers.Conv1D(filters=256, kernel_size=5, padding='same')(conv1)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.Activation('relu')(conv2)
#     conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(conv2)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.Activation('relu')(conv3)
#     conv3_shape = K.int_shape(conv3)
#     print(conv3_shape)
#     flatten = keras.layers.Flatten()(conv3)
#     flatten_shape = K.int_shape(flatten)
#     print(flatten_shape)

#     encoded = keras.layers.Dense(stt.FEATURES)(flatten)

#     # DECODER
#     h = keras.layers.Dense(flatten_shape[1] )(encoded)
#     h = keras.layers.Reshape( (conv3_shape[1], conv3_shape[2]) )( h )

#     conv3 = keras.layers.Conv1D(128, kernel_size=3, padding='same')(h)
#     conv3 = keras.layers.BatchNormalization()(conv3)
#     conv3 = keras.layers.Activation('relu')(conv3)
#     conv2 = keras.layers.Conv1D( filters=256, kernel_size=5, padding='same')(conv3)
#     conv2 = keras.layers.BatchNormalization()(conv2)
#     conv2 = keras.layers.Activation('relu')(conv2)
#     conv1 = keras.layers.Conv1D(filters=input_dim, kernel_size=8, padding='same')(conv2)
#     conv1 = keras.layers.BatchNormalization()(conv1)
#     conv1 = keras.layers.Activation(activation='relu')(conv1)
#     decoded = conv1
#     print("shape of decoded {}".format(K.int_shape(decoded)))

#     encoder = keras.Model(input_layer, encoded)
#     autoencoder = keras.Model(input_layer, decoded)
#     autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])    
#     reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=50, 
#                                                   min_lr=0.0001) 
#     model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose=1)
#     callbacks = [reduce_lr,model_checkpoint]
#     print(autoencoder.summary()) 
#     return callbacks, encoder, autoencoder
