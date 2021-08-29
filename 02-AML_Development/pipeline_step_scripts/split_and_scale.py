from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import os
import argparse

import pandas as pd
import numpy as np
from sklearn import preprocessing
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import joblib
from numpy.random import seed

#Parse input arguments
parser = argparse.ArgumentParser("Split raw data into train/test and scale appropriately")
parser.add_argument('--autoencoder_training_data', dest='autoencoder_training_data', required=True)
parser.add_argument('--autoencoder_testing_data', dest='autoencoder_testing_data', required=True)
parser.add_argument('--split_to_train_pipeline_data', dest='split_to_train_pipeline_data', required=True)

args, _ = parser.parse_known_args()
autoencoder_training_data = args.autoencoder_training_data
autoencoder_testing_data = args.autoencoder_testing_data
split_to_train_pipeline_data = args.split_to_train_pipeline_data

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

# Read input dataset to pandas dataframe
autoencoder_raw_datset = current_run.input_datasets['Autoencoder_Raw_Data']
raw_df = autoencoder_raw_datset.to_pandas_dataframe().astype(np.float64)

scaler = preprocessing.MinMaxScaler()

X_train = pd.DataFrame(scaler.fit_transform(raw_df),
                      columns=raw_df.columns,
                      index=raw_df.index)

# Save train data to both train and test (reflects the usage pattern in this sample. Note: test/train sets are typically distinct data).
os.makedirs(autoencoder_training_data, exist_ok=True)
os.makedirs(autoencoder_testing_data, exist_ok=True)
X_train.to_csv(os.path.join(autoencoder_training_data, 'autoencoder_training_data.csv'), index=False)
X_train.to_csv(os.path.join(autoencoder_testing_data, 'autoencoder_testing_data.csv'), index=False)

# Save scaler to PipelineData and outputs for record-keeping
os.makedirs('./outputs', exist_ok=True)
joblib.dump(scaler, './outputs/scaler.pkl')
os.makedirs(split_to_train_pipeline_data, exist_ok=True)
joblib.dump(scaler, os.path.join(split_to_train_pipeline_data, 'scaler.pkl'))