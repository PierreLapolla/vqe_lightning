import logging

from pedros import setup_logging


def configure_logging() -> None:
    setup_logging()

    # Route third-party logs through the app's root logger to avoid duplicate handlers
    for logger_name in ("lightning", "lightning.pytorch", "lightning.fabric", "wandb"):
        logger = logging.getLogger(logger_name)
        logger.handlers.clear()
        logger.propagate = True

    # Route Python warnings (warnings.warn) through logging so Rich can format them.
    logging.captureWarnings(True)
    warnings_logger = logging.getLogger("py.warnings")
    warnings_logger.handlers.clear()
    warnings_logger.propagate = True
