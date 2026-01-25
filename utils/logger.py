import logging
import os
import sys

def setup_logger(name, save_dir, distributed_rank=0, filename="log.txt", mode='a'):
    """
    Configures a logger to output to console and file.
    Only rank 0 logs to console/file usually.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Don't propagate to root logger
    logger.propagate = False
    
    if distributed_rank > 0:
        return logger
        
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    # Console Handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    # File Handler
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        fh = logging.FileHandler(os.path.join(save_dir, filename), mode=mode)
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        
    return logger
