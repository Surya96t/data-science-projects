import os
import sys
from sqlalchemy import create_engine
from dotenv import load_dotenv
from alzheimers.utils.logger import get_logger

# BASEDIR = os.path.abspath(os.path.dirname(__file__))
# DB_ENV_PATH = os.path.join(BASEDIR, 'alzheimers', 'data_ingestion', 'db.env')
# Load environment variables
load_dotenv(r'alzheimers\data_ingestion\db.env')    

logger = get_logger()

def get_db_engine():
    """ Create and return a SQLAlchemy database engine. """
    try:
        db_url = f"postgresql+psycopg2://{os.getenv('DB_USERNAME')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{os.getenv('DB_NAME')}"
        engine = create_engine(db_url)
        logger.info("Database connection established successfully.")
        return engine
    except Exception as e:
        logger.error(f"Error creating the database engine: {e}")
        sys.exit(1)
        