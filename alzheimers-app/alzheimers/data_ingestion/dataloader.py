import os
import sys
import pandas as pd
from alzheimers.utils.logger import get_logger
from alzheimers.data_ingestion.db_utils import get_db_engine
from alzheimers.utils.config_loader import load_config

# Get the logger
logger = get_logger()

# Load the configuration
config = load_config()

# Get the paths from YAML file
DATA_FOLDER = config["paths"]["raw_data_folder"]
DATA_FILE = config["paths"]["raw_data_filename"]

def fetch_and_save_data(folder=DATA_FOLDER):
    """ To fetch data from the database using SQLAlchemy
    and save it to a csv file"""
    
    # TODO: make the query dynamic
    query = "SELECT * FROM patient_records;"
    engine = get_db_engine()
    
    try:
        logger.info("Executing database query to fetch data.")
        df = pd.read_sql_query(query, engine)
        if df.empty:
            logger.warning("Query returned no data.")
            return
        
        os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
        filepath = os.path.join(DATA_FOLDER, DATA_FILE)
        df.to_csv(filepath, index=False)
        
        logger.info(f"Data successfully fetched and saved to {filepath}")
    except Exception as e:
        logger.error(f"Error fetching data: {e}")
        sys.exit(1)
    finally:
        engine.dispose()
        logger.info("Database connection closed.")

if __name__ == "__main__":
    fetch_and_save_data()

# import os 
# import sys
# import yaml
# from sqlalchemy import create_engine
# from dotenv import load_dotenv
# import pandas as pd
# from alzheimers.utils.logger import get_logger


# # Get the logger
# logger = get_logger()

# # Load environment variables
# load_dotenv(r'alzheimers\dataloader\db.env')

# # Load data paths from config file


# # Load the configuration
# config = load_config()

# # Get the paths from YAML file

# DATA_FOLDER = config["paths"]["data_folder"]
# DATA_FILE = config["paths"]["data_filename"]

# # Create a database engine
# def get_db_engine():
#     """ Create and return a SQLAlchemy database engine. """
#     try:
#         db_url = f"postgresql+psycopg2://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
#         engine = create_engine(db_url)
#         logger.info("Database connection established successfully.")
#         return engine
#     except Exception as e:
#         logger.error(f"Error creating the database engine: {e}")
#         sys.exit(1)

# def fetch_and_save_data(folder="data"):
#     """ To fetch data from the database using SQLAlchemy
#     and save it to a csv file"""
#     query = "SELECT * FROM alzheimers_data;"
#     engine = get_db_engine()
    
#     try:
#         logger.info("Executing database query to fetch data.")
#         df = pd.read_sql_query(query, engine)
#         if df.empty:
#             logger.warning("Query returned no data.")
#             return
        
#         os.makedirs(folder, exist_ok=True)  # Create folder if it doesn't exist
#         filepath = os.path.join(DATA_FOLDER, DATA_FILE)
#         df.to_csv(filepath, index=False)
        
#         logger.info(f"Data successfully fetched and saved to {filepath}")
#     except Exception as e:
#         logger.error(f"Error fetching data: {e}")
#         sys.exit(1)
#     finally:
#         engine.dispose()
#         logger.info("Database connection closed.")
    
    
# if __name__ == "__main__":
#     fetch_and_save_data()