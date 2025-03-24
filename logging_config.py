import logging
import os
from logging.handlers import RotatingFileHandler

def configure_logging():
    log_dir = os.environ.get("LOG_DIR", "/app/logs")
    os.makedirs(log_dir, exist_ok=True)

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(log_dir, 'app.log')

    # Console Handler (for Docker stdout)
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File Handler (for persistent logs)
    file_handler = RotatingFileHandler(
        log_file, 
        maxBytes=10 * 1024 * 1024,  # 10 MB per file
        backupCount=5  # Keep 5 backups
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(log_format))

    # Set up loggers
    logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler])

    loggers = {
        'train': logging.getLogger('ml_app.train'),
        'api': logging.getLogger('ml_app.api')
    }
    
    return loggers
