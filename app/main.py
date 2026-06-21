from fastapi import FastAPI, HTTPException
import optuna
import threading
from pydantic import BaseModel
import threading
import sys
from src.logger import configure_logger
from src.exception import MyException
from src.pipeline.prediction import prediction_pipeline
from src.pipeline.training import start_model_training
from src.utils.model_utils import load_model_from_mlflow
import datetime
logging = configure_logger()

app = FastAPI(title ="Nordex shift Performance")

model = None

@app.on_event("startup")
def load_model_on_startup():
    global model
    try:
        logging.info("loading model at startup...")
        model = load_model_from_mlflow()
        logging.info("model sucessfully loaded..")
    except Exception as e:
        logging.error("failed to load model from mlflow {e}")
        raise MyException(e, sys)

# schema
class ShiftInput(BaseModel):
   units_produced: int
   defect_count: int
   cycle_time_avg: float
   experience_level: int
   runtime_hours: float
   downtime_minutes: float
   maintenance_flag: int
   maintenance_downtime: float
   temperature: float
   humidity:float
   shift_duration:int
   day_of_week:int
   shift_name: str
   skill_category:str
   machine_status:str

class OptimizationInput(BaseModel):
    shift_name: str
    skill_category: str
    machine_status: str

    exp_range: tuple[int, int]
    downtime_range: tuple[float, float]
    defect_range: tuple[int, int]

    n_trials: int


@app.get("/")
def health_Check():
    return {"message": "Nordex shift performance API is Running"}

@app.post("/predict")
def predict_shift_effciency(input: ShiftInput):
    try:
        if model is None:
            raise Exception("model is not loaded from mlflow...")
        
        result = prediction_pipeline(input.dict(), model)

        return {
            "predicted_shift_efficiency_score": float(result[0])
        }
    except Exception as e:
        logging.error(f"prediction failed: {e}")
        raise HTTPException(status = 500, detail = str(e))


# Training
def retrain_pipeline():
    global model

    try:
        start_model_training()
        logging.info("Reloading the latest model from mlflow...")
        model = load_model_from_mlflow()

    except Exception as e:
        logging.error(f"re-training process failed...")
        raise MyException(e, sys)

@app.post("/retrain")
def retrain_model():
    thread = threading.Thread(target = retrain_pipeline)
    thread.start()

    return {
        "message": "Training started in the background, model will automatically update."
    }


@app.post("/optimize")
def optimize_shift(input: OptimizationInput):
    try:
        if model is None:
            raise Exception("model not loaded from mlflow")

        def objective(trial):

            experience_level = trial.suggest_int(
                "experience_level",
                input.exp_range[0],
                input.exp_range[1]
            )

            downtime_minutes = trial.suggest_float(
                "downtime_minutes",
                input.downtime_range[0],
                input.downtime_range[1]
            )

            defect_count = trial.suggest_int(
                "defect_count",
                input.defect_range[0],
                input.defect_range[1]
            )
            maintenance_downtime = trial.suggest_float("maintenance_downtime", 0, 120)

            units_produced = trial.suggest_int("units_produced", 600, 1200)
            maintenance_flag= trial.suggest_categorical("maintenance_flag", [0, 1])
            cycle_time_avg = trial.suggest_float("cycle_time_avg", 30, 40)
            temperature = trial.suggest_float("temperature", 18, 30)
            humidity = trial.suggest_float("humidity", 30, 70)
            runtime_hours = 7.5
            shift_duration = runtime_hours
            day_of_week = datetime.datetime.today().weekday()

            data = {
                "experience_level": experience_level,
                "downtime_minutes":downtime_minutes,
                "defect_count" : defect_count,
                "units_produced": units_produced,
                "maintenance_flag": maintenance_flag,
                "maintenance_downtime": maintenance_downtime,
                "cycle_time_avg": cycle_time_avg,
                "temperature": temperature,
                "humidity": humidity,
                "runtime_hours": runtime_hours,
                "shift_name": input.shift_name,
                "skill_category": input.skill_category,
                "machine_status": input.machine_status,
                "shift_duration": shift_duration,
                "day_of_week": day_of_week
            }

            pred = prediction_pipeline(data, model)

            return pred[0]
        
        study = optuna.create_study(direction= "maximize")

        study.optimize(objective, n_trials = input.n_trials)

        return {
            "best_parameters": study.best_params,
            "best_score": study.best_value,
            "top_trials": study.trials_dataframe()
            .sort_values("value", ascending=False)
            .head(10)
            .to_dict(orient = "records")
        }
    
    except Exception as e:
        logging.error(f"optimization failed: {e}")
        raise HTTPException(status_code = 500, detail=str(e))

        








