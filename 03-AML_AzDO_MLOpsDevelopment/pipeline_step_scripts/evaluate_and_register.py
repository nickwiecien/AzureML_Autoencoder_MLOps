from azureml.core import Run, Workspace, Datastore, Dataset
from azureml.core.model import Model
from azureml.data.datapath import DataPath
import pandas as pd
import os
import argparse
import shutil
import tensorflow as tf

# #Parse input arguments
parser = argparse.ArgumentParser("Evaluate autoencoder and register if more performant")
parser.add_argument('--autoencoder_training_outputs', type=str, required=True)

args, _ = parser.parse_known_args()
autoencoder_training_outputs = args.autoencoder_training_outputs

#Get current run
current_run = Run.get_context()

#Get associated AML workspace
ws = current_run.experiment.workspace

#Get default datastore
ds = ws.get_default_datastore()

#Get metrics associated with current parent run
metrics = current_run.parent.get_metrics()
current_model_mse = float(metrics['Mean Squared Error'])

#Get tags from current parent run
tags = current_run.parent.get_tags()

#Get training/testing datasets
training_dataset = current_run.input_datasets['Autoencoder_Training_Data']
testing_dataset = current_run.input_datasets['Autoencoder_Testing_Data']
formatted_datasets = [('Autoencoder_Training_Data', training_dataset), ('Autoencoder_Testing_Data', testing_dataset)]

# Get point to PipelineData from previous training step
training_step_pipeline_data = autoencoder_training_outputs

# Copy autoencoder training outputs to relative path for registration
relative_model_path = 'model_files'
current_run.upload_folder(name=relative_model_path, path=training_step_pipeline_data)

# Get current model from workspace
model_name = 'Autoencoder_PredMaintenance'
model_description = 'TF/Keras autoencoder for detecting anomalies in multi-variate IoT telemetry data'
model_list = Model.list(ws, name=model_name, latest=True)
first_registration = len(model_list)==0

updated_tags = {'Mean Squared Error': metrics['Mean Squared Error'], 'BuildId': tags['BuildId'], 'BuildUri': tags['BuildUri']}

#If no model exists register the current model
if first_registration:
    print('First model registration.')
    model = current_run.register_model(model_name, model_path='model_files', description=model_description, model_framework='Tensorflow/Keras', model_framework_version=tf.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
else:
    #If a model has been registered previously, check to see if current model 
    #performs better (lower MSE). If so, register it.
    print(dir(model_list[0]))
    if float(model_list[0].tags['Mean Squared Error']) > current_model_mse:
        print('New model performs better than existing model. Register it.')
        model = current_run.register_model(model_name, model_path='model_files', description=model_description, model_framework='Tensorflow/Keras', model_framework_version=tf.__version__, tags=updated_tags, datasets=formatted_datasets, sample_input_dataset = training_dataset)
    else:
        print('New model does not perform better than existing model. Cancel run.')
        current_run.parent.cancel()