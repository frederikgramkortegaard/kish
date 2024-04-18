import logging

logging.getLogger("matplotlib.font_manager").disabled = True


def setup_custom_logger(name):

    FORMAT = "[%(filename)s:%(lineno)s - %(funcName)20s() ] %(message)s"
    handler = logging.StreamHandler()
    handler.setFormatter(FORMAT)

    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    return logger