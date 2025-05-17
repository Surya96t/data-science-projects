import yaml
import sys
from alzheimers.utils.logger import get_logger

logger = get_logger()

def load_config(config_path=r"alzheimers\configs\dataConfig.yml"):
    """ Load a YAML configuration file. """
    try:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        logger.info(f"Configuration file {config_path} loaded successfully.")
        return config
    except Exception as e:
        logger.error(f"Error loading configuration file {config_path}: {e}")
        sys.exit(1)