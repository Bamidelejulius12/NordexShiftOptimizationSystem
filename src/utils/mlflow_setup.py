import dagshub
import mlflow
import os
from dotenv import load_dotenv
from config.constant import EXPERIMENT_NAME, dagshub_url, repo_owner, repo_name
load_dotenv(override = True)

# def setup_mlflow():
#     dagshub.init(
#         repo_owner='babatundejulius911', 
#         repo_name='NordexShiftOptimizationSystem', 
#         mlflow=True
#     )
#     mlflow.set_experiment(EXPERIMENT_NAME)

def setup_mlflow():
    dagshub_token = os.getenv("MLFLOW_TOKEN")
    if not dagshub_token:
        raise EnvironmentError("MLFLOW_TOKEN environment variable is not set")

    os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
    os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

    Dagshub_Url = "https://dagshub.com" 
    Repo_Owner = repo_owner
    Repo_Name = repo_name
    # Set up MLflow tracking URI
    mlflow.set_tracking_uri(f'{Dagshub_Url}/{Repo_Owner}/{Repo_Name}.mlflow')
    mlflow.set_experiment(EXPERIMENT_NAME)

  