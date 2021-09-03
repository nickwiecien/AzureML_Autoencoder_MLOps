#Import required packages
from azureml.core import Workspace,Environment
from azureml.core.compute import ComputeTarget, AmlCompute
from azureml.core.compute_target import ComputeTargetException

# Connect to AML Workspace
ws = Workspace.from_config()

from azureml.core.environment import Environment
from azureml.core.model import InferenceConfig, Model
from azureml.core.webservice import LocalWebservice
import os

model_name = os.getenv('MODEL_NAME', '')
model_version = os.getenv('MODEL_VERSION', '')


# Create inference configuration based on the environment definition and the entry script
myenv = Environment.get(ws, 'tf_keras_autoencoder_env')
inference_config = InferenceConfig(entry_script="scoring_scripts/score.py", environment=myenv)
model = Model(ws, name=model_name, version=model_version)

package2 = Model.package(ws, [model], inference_config, generate_dockerfile=True)
package2.wait_for_creation(show_output=True)

os.makedirs('./docker', exist_ok=True)

package2.save('./docker')