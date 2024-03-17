# debug_utils.py

import logging

DEBUG_MODE = True
logging.basicConfig(level=logging.DEBUG if DEBUG_MODE else logging.INFO)

def debug_log(message, data=None):
    if DEBUG_MODE:
        logging.debug(message)
        if data is not None:
            logging.debug("Debug Data: %s", data)