# AzureML_Autoencoder_MLOps

Supporting ML models in production requires management and maintenance throughout the entire model lifecycle. To this end, incorporating automation and MLOps best practices can dramatically reduce the effort required to ensure models are performing well, and delivering valuable insights.

This repo contains samples for development of an autoencoder (used for anomaly detection) from three different postures:

1.) Local Development - No MLOps

2.) Automated Retraining via Azure ML Pipelines - Partial MLOps

3.) Automated Retraining and Deployment via Azure ML & Azure DevOps - Full MLOps

The code for each development posture is located within its corresponding subdirectory.

The autoencoder training routine and source data were adapted from the [Microsoft Cloud Workshop - Predictive Maintenance for Remote Field Devices](https://github.com/microsoft/MCW-Predictive-Maintenance-for-remote-field-devices). The Infrastructure-as-Code and MLOps pipelines were adapted from [Microsoft's MLOps for Python Template Repository](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/mlops-python). 

![Autoencoder MLOps](img/banner.png?raw=true "Autoencoder-MLOps")


## Environment Setup
<b>Local Development - No MLOps</b>

The `01-Local_Development` directory contains a notebook (`autoencoder_training.ipynb`), pip requirements file (`requirements.txt`), and sample data located in the `sample_data` subdirectory. Recommend first creating a virtual environment and installing required pip packages before executing this notebook. Model outputs are saved locally.

<b>Automated Retraining via Azure ML Pipelines - Partial MLOps</b>

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


<b>Note:</b> Recommend running this notebook using an Azure Machine Learning compute instance using the preconfigured `Python 3.6 - AzureML` environment.

To build and run the sample pipeline contained in `SamplePipeline.ipynb` the following resources are required:
* Azure Machine Learning Workspace