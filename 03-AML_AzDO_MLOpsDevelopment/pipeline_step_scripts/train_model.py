from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse
import shutil

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from numpy.random import seed

import tensorflow as tf
from tensorflow.keras.layers import Input, Dropout
from keras.layers.core import Dense 
from keras.models import Model, Sequential, load_model
from keras import regularizers
from keras.models import model_from_json


#Parse input arguments
parser = argparse.ArgumentParser("Train TF/Keras autoencoder model")
parser.add_argument('--train_to_evaluate_pipeline_data', dest='train_to_evaluate_pipeline_data', required=True)
parser.add_argument('--split_to_train_pipeline_data', dest='split_to_train_pipeline_data', required=True)
parser.add_argument('--num_epochs', type=int, required=True)
parser.add_argument('--batch_size', type=int, required=True)

args, _ = parser.parse_known_args()
train_to_evaluate_pipeline_data = args.train_to_evaluate_pipeline_data
split_to_train_pipeline_data = args.split_to_train_pipeline_data
num_epochs = args.num_epochs
batch_size = args.batch_size

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
X_train_dataset = current_run.input_datasets['Autoencoder_Training_Data']
X_train = X_train_dataset.to_pandas_dataframe().astype(np.float64)
X_test_dataset = current_run.input_datasets['Autoencoder_Testing_Data']
X_test = X_test_dataset.to_pandas_dataframe().astype(np.float64)

#Train autoencoder
act_func = 'elu'

input = Input(shape=(X_train.shape[1],))
x = Dense(100,activation=act_func, kernel_initializer='glorot_uniform', kernel_regularizer=regularizers.l2(0.0))(input)
x = Dense(50,activation=act_func, kernel_initializer='glorot_uniform')(x)
encoder = Dense(20,activation=act_func, kernel_initializer='glorot_uniform', name='feature_vector')(x)
x = Dense(50,activation=act_func, kernel_initializer='glorot_uniform')(encoder)
x = Dense(100,activation=act_func, kernel_initializer='glorot_uniform')(x)
output = Dense(X_train.shape[1],activation=act_func, kernel_initializer='glorot_uniform')(x)

model = Model(input, output)
model.compile(loss='mse',optimizer='adam')

encoder_model = Model(inputs=model.input, outputs=model.get_layer('feature_vector').output)
encoder_model.compile(loss='mse',optimizer='adam')

history=model.fit(np.array(X_train),np.array(X_train),
                  batch_size=batch_size, 
                  epochs=num_epochs,
                  validation_split=0.05,
                  verbose = 1)

# Save model to outputs for record-keeping
os.makedirs('./outputs', exist_ok=True)
model.save('./outputs/anomaly_detection_encoder_model.h5')

# Save model to pipeline_data for use in evaluation/registration step
os.makedirs(train_to_evaluate_pipeline_data, exist_ok=True)
model.save(os.path.join(train_to_evaluate_pipeline_data, 'anomaly_detection_encoder_model.h5'))
shutil.copyfile(os.path.join(split_to_train_pipeline_data, 'scaler.pkl'), os.path.join(train_to_evaluate_pipeline_data, 'scaler.pkl'))


# Add MSE metric to run and parent run
current_run.log(name='Mean Squared Error', value=history.history['loss'][-1])
current_run.parent.log(name='Mean Squared Error', value=history.history['loss'][-1])
current_run.log_list(name='Training Loss (MSE)', value=history.history['loss'])
current_run.parent.log_list(name='Training Loss (MSE)', value=history.history['loss'])
current_run.log_list(name='Validation Loss (MSE)', value=history.history['val_loss'])
current_run.parent.log_list(name='Validation Loss (MSE)', value=history.history['val_loss'])

#Generate training loss plot and add to run and parent run
fig, ax = plt.subplots(1,1, figsize=(20,5))

ax.set_title('Training Loss')
plt.plot(history.history['loss'],
         'r',
         label='Training loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.002])
current_run.log_image(name='Training Loss (MSE) Plot', plot=plt)
current_run.parent.log_image(name='Training Loss (MSE) Plot', plot=plt)

#Generate validation loss plot and add to run and parent run
fig, ax = plt.subplots(1,1, figsize=(20,5))

ax.set_title('Validation loss')
plt.plot(history.history['val_loss'],
         'b',
         label='Validation loss')
plt.legend(loc='upper right')
plt.xlabel('Epochs')
plt.ylabel('Loss, [mse]')
plt.ylim([0,.002])
current_run.log_image(name='Validation Loss (MSE) Plot', plot=plt)
current_run.parent.log_image(name='Validation Loss (MSE) Plot', plot=plt)


