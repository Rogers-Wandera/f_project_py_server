import logging
from datetime import datetime
import os

class Logger:
    def __init__(self, log_file="log.txt"):
        parentdir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_directory = os.path.join(parentdir, "logs")

        # Ensure that the 'logs' directory exists
        if not os.path.exists(log_directory):
            os.makedirs(log_directory, exist_ok=True)

        logpath = os.path.join(log_directory, log_file)

        self.logger = logging.getLogger("Logger")
        self.logger.setLevel(logging.INFO)
        
        file_handler = logging.FileHandler(logpath, mode='a')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)

        self.logger.addHandler(file_handler)
    
    def logaccess(self, username, action):
        log_message = f"{datetime.now()} - User: {username} - Action: {action}"
        self.logger.info(log_message)
