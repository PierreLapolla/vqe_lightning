import logging

from pedros import setup_logging


def configure_logging() -> None:
    setup_logging()

    for logger_name in ("lightning", "lightning.pytorch", "lightning.fabric", "wandb"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True

    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.propagate = True
