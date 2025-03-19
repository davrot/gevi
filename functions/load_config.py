import json
import os
import logging

from jsmin import jsmin  # type:ignore


def load_config(mylogger: logging.Logger, filename: str = "config.json") -> dict:
    mylogger.info("loading config file")
    if os.path.isfile(filename) is False:
        mylogger.info(f"{filename} is missing")

    with open(filename, "r") as file:
        config = json.loads(jsmin(file.read()))

    return config
