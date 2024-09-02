import sys
from loguru import logger

LOGGING_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{process.name}:{thread.name}</cyan> | "
    "<blue>{name}:{function}:{line}</blue> - "
    "<level>{message}</level>"
)


def configure_logger():
    logger.remove()  # Remove the default logger configuration
    logger.add(sys.stdout, colorize=True, format=LOGGING_FORMAT)
    logger.add("combined_rag_evaluator.log", rotation="500 MB", retention="10 days", format=LOGGING_FORMAT,
               compression="zip")
