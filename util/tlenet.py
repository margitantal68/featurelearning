
import tensorflow.keras as keras
import tensorflow as tf
import numpy as np 
import time


def build_tlenet(input_shape, nb_classes, file_path):
    input_layer = keras.layers.Input(input_shape)
    
    conv_1 = keras.layers.Conv1D(filters=5,kernel_size=5,activation='relu', padding='same')(input_layer)
    conv_1 = keras.layers.MaxPool1D(pool_size=2)(conv_1)
    
    conv_2 = keras.layers.Conv1D(filters=20, kernel_size=5, activation='relu', padding='same')(conv_1)
    conv_2 = keras.layers.MaxPool1D(pool_size=4)(conv_2)
    
    # they did not mention the number of hidden units in the fully-connected layer
    # so we took the lenet they referenced 
    
    flatten_layer = keras.layers.Flatten()(conv_2)
    fully_connected_layer = keras.layers.Dense(500,activation='relu')(flatten_layer)
    
    output_layer = keras.layers.Dense(nb_classes,activation='softmax')(fully_connected_layer)
    
    model = keras.models.Model(inputs=input_layer,outputs=output_layer)
    
    model.compile(optimizer=keras.optimizers.Adam(lr=0.01,decay=0.005),
                    loss='categorical_crossentropy', metrics=['accuracy'])
    
    model_checkpoint = keras.callbacks.ModelCheckpoint(filepath=file_path, monitor='loss', save_best_only=True, verbose = 1)
    callbacks = [model_checkpoint]

    return callbacks, model