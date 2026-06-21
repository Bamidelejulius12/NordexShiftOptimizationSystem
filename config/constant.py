import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
database_path = os.path.abspath(os.path.join(BASE_DIR, 'ShiftData.db'))
SCHEMA_PATH =  os.path.abspath(os.path.join(BASE_DIR, 'config', 'schema.yml'))
feature_artifact = os.path.abspath(os.path.join(BASE_DIR, 'artifacts', 'features', 'feature_engineering.csv'))

LOG_DIR = "logs"
LOG_FILE = "app.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3
EXPERIMENT_NAME = "Nordex_Shift_Optimization_Production_Experiment_2"
registered_model_name = "shift_optimization_production_model"

log_folder_path = os.path.join(os.getcwd(), LOG_DIR)
os.makedirs(log_folder_path, exist_ok=True)

log_file_path = os.path.join(log_folder_path, LOG_FILE)

target_column = "shift_efficiency_score"
dagshub_url = "https://dagshub.com"
repo_owner = "babatundejulius911"
repo_name = "NordexShiftOptimizationSystem"