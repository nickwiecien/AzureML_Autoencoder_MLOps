from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
from sklearn import preprocessing
import numpy as np

#Parse input arguments
parser = argparse.ArgumentParser("Get data from ADLS Gen2 and register in AML workspace")
parser.add_argument('--autoencoder_raw_dataset', dest='autoencoder_raw_dataset', required=True)

args, _ = parser.parse_known_args()
autoencoder_raw_dataset = args.autoencoder_raw_dataset

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Connect to ADLS Gen2 datastore
ds = Datastore.get(ws, 'adlsgen2datastore')

#Read all raw data from ADLS Gen2
csv_paths = [(ds, 'normal/*')]
raw_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
raw_df = raw_ds.to_pandas_dataframe().astype(np.float64)

#Make directory on mounted storage
os.makedirs(autoencoder_raw_dataset, exist_ok=True)

#Upload modified dataframe
raw_df.to_csv(os.path.join(autoencoder_raw_dataset, 'autoencoder_raw_data.csv'), index=False)
