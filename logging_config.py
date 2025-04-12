import logging
import os
from logging.handlers import RotatingFileHandler

def configure_logging():
    # Use ./logs directory in project root by default, or from environment variable
    log_dir = os.environ.get("LOG_DIR", os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs"))
    
    try:
        os.makedirs(log_dir, exist_ok=True)
    except PermissionError:
        # Fallback to system temp directory if we can't write to the desired location
        import tempfile
        log_dir = os.path.join(tempfile.gettempdir(), "app_logs")
        os.makedirs(log_dir, exist_ok=True)
        logging.warning(f"Could not create logs in default directory, using temp directory: {log_dir}")

    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    log_file = os.path.join(log_dir, 'app.log')

    # Clear any existing handlers
    logging.root.handlers = []

    # Console Handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter(log_format))

    # File Handler
    try:
        file_handler = RotatingFileHandler(
            log_file, 
            maxBytes=10 * 1024 * 1024,  # 10 MB per file
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter(log_format))
    except PermissionError as e:
        logging.error(f"Could not create log file at {log_file}: {e}")
        file_handler = None

    # Configure root logger
    handlers = [console_handler]
    if file_handler:
        handlers.append(file_handler)
        
    logging.basicConfig(
        level=logging.INFO,
        handlers=handlers,
        force=True  # Override any existing handlers
    )

    # Create and return specific loggers
    loggers = {
        'train': logging.getLogger('ml_app.train'),
        'api': logging.getLogger('ml_app.api')
    }
    
    return loggers