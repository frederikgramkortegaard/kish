import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def setup_custom_logger(name):
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(levelname)s - filename: %(filename)s - %(module)s - function: %(funcName)s - line: %(lineno)s - %(message)s"
    )

    handler = logging.StreamHandler()
    handler.setFormatter(formatter)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger