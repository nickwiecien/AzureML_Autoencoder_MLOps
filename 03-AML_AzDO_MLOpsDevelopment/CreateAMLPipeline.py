
# ## Azure ML Pipeline - Parameterized Input Dataset
# This notebook demonstrates creation & execution of an Azure ML pipeline designed to accept a parameterized input reflecting the location of a file in the Azure ML default datastore to be initially registered as a tabular dataset and subsequently processed. This notebook was built as part of a larger solution where files were moved from a blob storage container to the default AML datastore via Azure Data Factory.

from azureml.core import Workspace, Experiment, Datastore, Environment, Dataset
from azureml.core.compute import ComputeTarget, AmlCompute, DataFactoryCompute
from azureml.core.compute_target import ComputeTargetException
from azureml.core.runconfig import RunConfiguration
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core.runconfig import DEFAULT_CPU_IMAGE
from azureml.pipeline.core import Pipeline, PipelineParameter, PipelineData
from azureml.pipeline.steps import PythonScriptStep
from azureml.pipeline.core import PipelineParameter, PipelineData
from azureml.data.output_dataset_config import OutputTabularDatasetConfig, OutputDatasetConfig, OutputFileDatasetConfig
from azureml.data.datapath import DataPath
from azureml.data.data_reference import DataReference
from azureml.data.sql_data_reference import SqlDataReference
from azureml.pipeline.steps import DataTransferStep
import logging
import os

# ### Connect to Azure ML Workspace, Provision Compute Resources, and get References to Datastores
# Connect to workspace using config associated config file. Get a reference to you pre-existing AML compute cluster or provision a new cluster to facilitate processing. Finally, get references to your default blob datastore.

# Connect to AML Workspace
subscription_id = os.getenv("SUBSCRIPTION_ID", default="")
resource_group = os.getenv("RESOURCE_GROUP", default="")
workspace_name = os.getenv("WORKSPACE_NAME", default="")
workspace_region = os.getenv("WORKSPACE_REGION", default="")

try:
    # ws = Workspace.from_config()
    ws = Workspace(subscription_id=subscription_id, 
                   resource_group=resource_group, 
                   workspace_name=workspace_name)
    print("Workspace configuration succeeded. Skip the workspace creation steps below")
except:
    print("Workspace does not exist. Creating workspace")
    ws = Workspace.create(name=workspace_name, subscription_id=subscription_id, resource_group=resource_group,
                            location=workspace_region, create_resource_group=True, sku='enterprise', exist_ok=True)
#Select AML Compute Cluster
cpu_cluster_name = 'cpucluster'

# Verify that cluster does not exist already
try:
    cpu_cluster = ComputeTarget(workspace=ws, name=cpu_cluster_name)
    print('Found an existing cluster, using it instead.')
except ComputeTargetException:
    compute_config = AmlCompute.provisioning_configuration(vm_size='STANDARD_D3_V2',
                                                           min_nodes=0,
                                                           max_nodes=1)
    cpu_cluster = ComputeTarget.create(ws, cpu_cluster_name, compute_config)
    cpu_cluster.wait_for_completion(show_output=True)
    
#Get default datastore
default_ds = ws.get_default_datastore()

# ### Create Run Configuration
# The RunConfiguration defines the environment used across all python steps. You can optionally add additional conda or pip packages to be added to your environment. [More details here.](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.conda_dependencies.condadependencies?view=azure-ml-py)
# ~~~
# run_config.environment.python.conda_dependencies = CondaDependencies.create(conda_packages=['requests'])
# run_config.environment.python.conda_dependencies.add_pip_package('azureml-opendatasets')
# ~~~

run_config = RunConfiguration()
run_config.docker.use_docker = True
run_config.environment = Environment(name='tf_keras_autoencoder_env')
run_config.environment.docker.base_image = DEFAULT_CPU_IMAGE
run_config.environment.python.conda_dependencies = CondaDependencies.create()
run_config.environment.python.conda_dependencies.set_pip_requirements([
    'requests==2.26.0',
    'pandas==0.25.3',
    'numpy==1.19.2',
    'scikit-learn==0.22.2.post1',
    'joblib==0.14.1',
    'h5py==3.1.0',
    'tensorflow==2.6.0',
    'keras==2.6.0',
    'azureml-defaults==1.33.0',
    'matplotlib'
])
run_config.environment.python.conda_dependencies.set_python_version('3.8.10')
#Register environment for reuse 
run_config.environment.register(ws)

# ### Define Output Datasets
# Below we define the configuration for datasets that will be passed between steps in our pipeline. Note, in all cases we specify the datastore that should hold the datasets and whether they should be registered following step completion or not. This can optionally be disabled by removing the register_on_complete() call. upload_file_dataset is intended to hold the data within an uploaded CSV file and processed_dataset will contain our uploaded data post-processing.

autoencoder_raw_data = OutputFileDatasetConfig(name='Autoencoder_Raw_Data', destination=(default_ds, 'autoencoder_raw_data/{run-id}')).read_delimited_files().register_on_complete(name='Autoencoder_Raw_Data')
autoencoder_training_data = OutputFileDatasetConfig(name='Autoencoder_Training_Data', destination=(default_ds, 'autoencoder_training_data/{run-id}')).read_delimited_files().register_on_complete(name='Autoencoder_Training_Data')
autoencoder_testing_data = OutputFileDatasetConfig(name='Autoencoder_Testing_Data', destination=(default_ds, 'autoencoder_testing_data/{run-id}')).read_delimited_files().register_on_complete(name='Autoencoder_Testing_Data')

# ### Define Pipeline Data
# Fill in description of pipeline data here...

split_to_train_pipeline_data = PipelineData(name='Autoencoder_SplitScale_Outputs', datastore=default_ds)
train_to_evaluate_pipeline_data = PipelineData(name='Autoencoder_Training_Outputs', datastore=default_ds)

# ### Define Pipeline Parameters
# PipelineParameter objects serve as variable inputs to an Azure ML pipeline and can be specified at runtime. Below we specify a pipeline parameter object uploaded_file_path_param which will be used to define the locations of uploaded data inside the default Azure ML Blob datastore. Multiple pipeline parameters can be created and used.

num_epochs = PipelineParameter(name='num_epochs', default_value=10)
batch_size = PipelineParameter(name='batch_size', default_value=10)

# ### Define Pipeline Steps
# The pipeline below consists of two steps - one step to gather and register the uploaded file in the AML datastore, and a secondary step to consume and process this registered dataset. Also, any PipelineParameters defined above can be passed to and consumed within these steps.

#Get raw data from registered ADLS Gen2 datastore
#Register tabular dataset after retrieval
get_data_step = PythonScriptStep(
    name='Get Data from ADLS Gen2',
    script_name='get_data.py',
    arguments =['--autoencoder_raw_data', autoencoder_raw_data],
    outputs=[autoencoder_raw_data],
    compute_target=cpu_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

split_scale_step = PythonScriptStep(
    name='Split and Scale Raw Data',
    script_name='split_and_scale.py',
    arguments =['--autoencoder_training_data', autoencoder_training_data,
                '--autoencoder_testing_data', autoencoder_testing_data,
                '--split_to_train_pipeline_data', split_to_train_pipeline_data],
    inputs=[autoencoder_raw_data.as_input(name='Autoencoder_Raw_Data')],
    outputs=[autoencoder_training_data, autoencoder_testing_data, split_to_train_pipeline_data],
    compute_target=cpu_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

#Train autoencoder using raw data as an input
#Raw data will be preprocessed and registered as train/test datasets
#Scaler and train autoencoder will be saved out
train_model_step = PythonScriptStep(
    name='Train TF/Keras Autoencoder',
    script_name='train_model.py',
    arguments =[
                '--train_to_evaluate_pipeline_data', train_to_evaluate_pipeline_data,
                '--split_to_train_pipeline_data', split_to_train_pipeline_data,
                '--num_epochs', num_epochs,
                '--batch_size', batch_size],
    inputs=[autoencoder_training_data.as_input(name='Autoencoder_Training_Data'),
            autoencoder_testing_data.as_input(name='Autoencoder_Testing_Data'),
            split_to_train_pipeline_data.as_input('Autoencoder_SplitScale_Outputs')
           ],
    outputs=[train_to_evaluate_pipeline_data],
    compute_target=cpu_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

#Evaluate and register model here
#Compare metrics from current model and register if better than current
#best model
evaluate_and_register_step = PythonScriptStep(
    name='Evaluate and Register Autoencoder',
    script_name='evaluate_and_register.py',
    arguments=['--autoencoder_training_outputs', train_to_evaluate_pipeline_data],
    inputs=[autoencoder_training_data.as_input(name='Autoencoder_Training_Data'),
            autoencoder_testing_data.as_input(name='Autoencoder_Testing_Data'),
            train_to_evaluate_pipeline_data.as_input('Autoencoder_Training_Outputs')],
    compute_target=cpu_cluster,
    source_directory='./pipeline_step_scripts',
    allow_reuse=False,
    runconfig=run_config
)

# ### Create Pipeline
# Create an Azure ML Pipeline by specifying the steps to be executed. Note: based on the dataset dependencies between steps, exection occurs logically such that no step will execute unless all of the necessary input datasets have been generated.

pipeline = Pipeline(workspace=ws, steps=[get_data_step, split_scale_step, train_model_step, evaluate_and_register_step])

# ### Publish Pipeline
# Create a published version of your pipeline that can be triggered via an authenticated REST API request.

build_id = os.getenv('BUILD_BUILDID', default='1')
pipeline_name = os.getenv("PIPELINE_NAME", default="autoencoder-training-registration-pipeline")

published_pipeline = pipeline.publish(name = pipeline_name,
                                        version=build_id,
                                     description = 'Pipeline to load/register IoT telemetry data from ADLS Gen2, train a Tensorflow/Keras autoencoder for anomaly detection, and register the trained model if it performs better than the current best model.',
                                     continue_on_step_failure = False)


