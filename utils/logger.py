import logging
import sys

class Logging():
    """
        This class supports Logging the messages to the passed filepath.
    """
    def __init__(self, experiment_id, filepath):
        self.logger = logging.getLogger(experiment_id)
        c_handler = logging.FileHandler(filepath)
        f_handler = logging.StreamHandler()

        # Set logging level
        self.logger.setLevel(logging.DEBUG)

        # Create formatters and add it to handlers
        c_format = logging.Formatter(
            # '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            "[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
        )
        f_format = logging.Formatter(
            "[%(asctime)s] p%(process)s {%(filename)s:%(lineno)d} %(levelname)s - %(message)s"
        )
        c_handler.setFormatter(c_format)
        f_handler.setFormatter(f_format)

        # Add handlers to the logger
        self.logger.addHandler(c_handler)
        self.logger.addHandler(f_handler)

        # sys.stdout = self.logger.info
        # sys.stderr = self.logger.info