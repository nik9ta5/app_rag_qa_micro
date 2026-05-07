import logging
import os

def create_logger(dir_for_save: str, log_file: str):
    
    full_path2save = f"{dir_for_save}/{log_file}"

    logger = logging.getLogger("LogApp")
    logger.setLevel(logging.INFO) 

    if logger.handlers:
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    file_handler = logging.FileHandler(full_path2save, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.DEBUG) 
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger