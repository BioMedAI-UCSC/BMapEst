"""
    python3 optimize_maps.py \
        --exp mcflash \
        --data data/brainweb/ \
        --optimize_csf \
        --kspace_loss \
        --linear_comb_optim \
        -s mc_flash
"""

import os

from utils.logger import Logging
from utils.argparser import parse_args
from utils.util import log_args
from src.pipeline import Pipeline

def setup_logger(experiment_id, filepath):
    """
        Method to setup the logger globally.
        Parameters
        ----------
            experiment_id: str
                Experiment ID passed in Args
            filepath: str
                Output file of the logger
        Return
        ------
            None
    """
    logger = Logging(experiment_id, filepath)
    globals()["logger"] = logger
    return logger

def optimize_maps(args, logger):
    """
        Method to perform probability map optimization
        Parameter
        ---------
            args: argparse.ArgumentParser
                Argparser object
            logger: logging
                Python's logging object
        Return
        ------
            None
    """
    pipeline = Pipeline(args, logger)
    pipeline.build_pipeline()
    pipeline.optimize_maps()

def main(args):
    """
        Main function executions starts here.
        Parameters
        ----------
            args: argparse.ArgumentParser
                  Argparser object
        Return
        ------
            None
    """
    logger = setup_logger(
        args.exp, os.path.join(args.output, "logs.txt")
    ).logger
    log_args(args, logger)
    logger.info(f"Experiment: {args.exp}")
    optimize_maps(args, logger)
    logger.info("Optimization finished!")

if __name__=="__main__":
    main(parse_args())
