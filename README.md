# AzureML_Autoencoder_MLOps

Supporting ML models in production requires management and maintenance throughout the entire model lifecycle. To this end, incorporating automation and MLOps best practices can dramatically reduce the effort required to ensure models are performing well, and delivering valuable insights.

This repo contains samples for development of an autoencoder (used for anomaly detection) from three different postures:

* <b>Local Development - No MLOps</b>
* <b>Automated Retraining via Azure ML Pipelines - Partial MLOps</b>
* <b>Automated Retraining and Deployment via Azure ML & Azure DevOps - Full MLOps</b>

The code for each development posture is located within its corresponding subdirectory.

The autoencoder training routine and source data were adapted from the [Microsoft Cloud Workshop - Predictive Maintenance for Remote Field Devices](https://github.com/microsoft/MCW-Predictive-Maintenance-for-remote-field-devices). 

The Infrastructure-as-Code and MLOps pipelines were adapted from [Microsoft's MLOps for Python Template Repository](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/mlops-python). 

![Autoencoder MLOps](img/banner.png?raw=true "Autoencoder-MLOps")


## Environment Setup
### Local Development - No MLOps

The `01-Local_Development` directory contains a notebook `autoencoder_training.ipynb`, pip requirements file `requirements.txt`, and sample data located in the `sample_data` subdirectory. Recommend first creating a virtual environment and installing required pip packages before executing this notebook. Model outputs are saved locally.


### Automated Retraining via Azure ML Pipelines - Partial MLOps ###

To build and run the sample pipeline contained in `CreateAMLPipeline.ipynb` the following resources are required:
* Azure Machine Learning Workspace
* ADLS Gen 2 Storage Account
* ADLS Gen 2 File System with sample data from `01-Local_Development/sample_data` organized in the following structure:
~~~
iot-telemetry/
├─ normal/
│  ├─ Normal.csv
├─ gradual/
│  ├─ Gradual.csv
├─ immediate/
│  ├─ Immediate.csv
~~~
Prior to creating the automated retraining pipeline, register the ADLS Gen 2 Storage Account as a datastore in the AML workspace [using the steps described here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-access-data#azure-data-lake-storage-generation-2). 

To create the automated retraining pipeline, run the `CreateAMLPipeline.ipynb` notebook. <i>Note:</i> it is recommended to run this notebook from an Azure Machine Learning compute instance using the preconfigured `Python 3.8 - AzureML` environment. The `ScoringDevelopment.ipynb` notebook showcases local development and debugging of model endpoint deployments.

### Automated Retraining and Deployment via Azure ML & Azure DevOps - Full MLOps ###

To build and run the MLOps pipelines contained in the `03-AML_AzDO_MLOpsDevelopment/.pipelines` directory the following resources are required:
* Azure DevOps Project

To deploy a new production resource group and workspace the follow the linked steps below from the MLOps for Python Template Repo:
* [Setup Azure DevOps](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#setting-up-azure-devops)
* [Connect Your Code](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#get-the-code) - Here you will connect this repo or your own fork
* [Create a Variable Group for your Pipeline](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-a-variable-group-for-your-pipeline)
* [Create a DevOps Service Connection](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#get-the-code)
* [Create the IaC Pipeline](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-the-iac-pipeline) - Use `03-AML_AzDO_MLOpsDevelopment/.pipelines/iac-create-environment-pipeline-arm.yml`
* [Create a DevOps Service Connection for the ML Workspace](https://github.com/microsoft/MLOpsPython/blob/master/docs/getting_started.md#create-an-azure-devops-service-connection-for-the-azure-ml-workspace)
* Setup the continuous integration pipeline using `03-AML_AzDO_MLOpsDevelopment/.pipelines/publish_autoencoder_train_register.yml`
* Setup the continuous deployment pipeline using `03-AML_AzDO_MLOpsDevelopment/.pipelines/autoencoder-real-time-endpoint-cd.yml`

<i>Note:</i> Currently the connection to ADLS Gen 2 as a datastore in the production environment is not automated. Prior to running these CI/CD pipelines the steps for establishing that connection need to be repeated in the new AML workspace.