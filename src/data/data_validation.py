import pandas as pd
import numpy as np
import sys
from src.logger import configure_logger
from src.exception import MyException
from src.data.data_ingestion import load_data

logging = configure_logger()

class DataValidator:
    """
    a class that basically checkes and validate our data,
    to know, how of quality the data is, and if there is any missing values, or so on.
    """

    def __init__(self, data: pd.DataFrame):
        self.data = data
        logging.info("data validator has been initialized..")

    def check_data_if_empty(self):
        """
            this function checkes if the data is empty or not, and if it is empty, it will raise an exception.
        """
        try:
            logging.info("checking if the data is empty or not...")
            if self.data.empty:
                logging.error("data is empty, please check your data source...")
                raise MyException("data is empty, please check your data source...", sys)
            else:
                logging.info("data is not empty, proceeding to the next step...")
        except Exception as e:
            logging.error(f"Error occurred while checking if the data is empty: {e}")
            raise MyException(e, sys)
    
    def checking_missing_values(self):
        """
        this function checks for missing values.
        """
        missing_values = self.data.isna().sum()

        if missing_values.sum() > 0:
            logging.warning(f"missing values detected\n {missing_values}")
        else:
            logging.info("no missing values detected, proceeding to the next step...")
        return missing_values

    def checking_for_duplicates(self):
        """
        this function checks if the data has duplicates.
        """
        try:
            duplicates = self.data.duplicated().sum()

            if duplicates > 0:
                logging.warning(f"duplicates detected with the total number of : {duplicates}")
            else:
                logging.info("no duplicates detected, proceeding to the next step...")
        except Exception as e:
            logging.error(f"Error occurred while checking for duplicates: {e}")
            raise MyException(e, sys)
    
def starting_datavalidation(data: pd.DataFrame):
    try:
        logging.info("the data validator pipeline has started...")
        validator_engine = DataValidator(data)
        validator_engine.check_data_if_empty()
        validator_engine.checking_missing_values()
        validator_engine.checking_for_duplicates()

        logging.info("data validation completed successfully...")
        logging.info(data.head())

        return data
    except Exception as e:
        logging.error(f"Error occurred during data validation: {e}")
        raise MyException(e, sys)


data = load_data()
starting_datavalidation(data)