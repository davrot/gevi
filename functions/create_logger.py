import logging
import datetime
import os


def create_logger(
    save_logging_messages: bool, display_logging_messages: bool, log_stage_name: str
):
    now = datetime.datetime.now()
    dt_string_filename = now.strftime("%Y_%m_%d_%H_%M_%S")

    logger = logging.getLogger("MyLittleLogger")
    logger.setLevel(logging.DEBUG)

    if save_logging_messages:
        time_format = "%b %d %Y %H:%M:%S"
        logformat = "%(asctime)s %(message)s"
        file_formatter = logging.Formatter(fmt=logformat, datefmt=time_format)
        os.makedirs("logs_" + log_stage_name, exist_ok=True)
        file_handler = logging.FileHandler(
            os.path.join("logs_" + log_stage_name, f"log_{dt_string_filename}.txt")
        )
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

    if display_logging_messages:
        time_format = "%H:%M:%S"
        logformat = "%(asctime)s %(message)s"
        stream_formatter = logging.Formatter(fmt=logformat, datefmt=time_format)

        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(logging.INFO)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    return logger
