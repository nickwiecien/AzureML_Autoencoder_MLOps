#Import required packages
from azureml.core import Workspace,Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException
import os

# Connect to AML Workspace
subscription_id = os.getenv("SUBSCRIPTION_ID", default="")
resource_group = os.getenv("RESOURCE_GROUP", default="")
workspace_name = os.getenv("WORKSPACE_NAME", default="")
workspace_region = os.getenv("WORKSPACE_REGION", default="")

ws = Workspace(subscription_id=subscription_id, 
                   resource_group=resource_group, 
                   workspace_name=workspace_name)

from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import LocalWebservice

model_name = os.getenv('MODEL_NAME', '')
model_version = os.getenv('MODEL_VERSION', '')

# Create inference configuration based on the environment definition and the entry script
myenv = Environment.get(ws, 'tf_keras_autoencoder_env')
inference_config = InferenceConfig(entry_script="scoring/score.py", environment=myenv)
model = Model(ws, name=model_name, version=model_version)

package2 = Model.package(ws, [model], inference_config, generate_dockerfile=True)
package2.wait_for_creation(show_output=True)

os.makedirs('./docker', exist_ok=True)

package2.save('./docker')