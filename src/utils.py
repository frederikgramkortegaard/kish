""" Various methods that are used throughout the project. """

import os
import sys
import pickle
import time
import json

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.log import setup_custom_logger
from src.classes import Reinforcement, Network

logger = setup_custom_logger(__name__)


def save_network_output(
    path: str, input: Network.TrainingOutput, custom_name: str = None
):
    """Save the output of a network to a file"""

    logger.info(f"Saving network output")

    # Ensure that path is a dir
    if not os.path.isdir(path):
        os.mkdir(path)

    if custom_name is not None:
        with open(os.path.join(path, f"{custom_name}.pkl"), "wb") as f:
            pickle.dump(input, f)
    else:
        # Save the output
        with open(os.path.join(path, f"network_output-{time.time()}.pkl"), "wb") as f:
            pickle.dump(input, f)


def save_reinforcement_agent_output(
    path: str, input: Reinforcement.TrainingOutput, custom_name: str = None
):
    """Save the output of a reinforcement agent to a file"""

    logger.info(f"Saving reinforcement agent output")

    # Ensure that path is a dir
    if not os.path.isdir(path):
        os.mkdir(path)

    if custom_name is not None:
        with open(os.path.join(path, f"{custom_name}.pkl"), "wb") as f:
            pickle.dump(input, f)
    else:
        # Save the output
        with open(
            os.path.join(path, f"reinforcement_agent_output-{time.time()}.pkl"), "wb"
        ) as f:
            pickle.dump(input, f)


def save_network_test_results(
    path: str, input: Network.TestingOutput, custom_name: str = None
):
    """Save the results of a network test to a file"""

    logger.info(f"Saving network test results")

    # Ensure that path is a dir
    if not os.path.isdir(path):
        os.mkdir(path)

    input = input.__dict__

    if custom_name is not None:
        with open(os.path.join(path, f"{custom_name}.json"), "w") as f:
            json.dump(input, f)
    else:
        json.dump(
            input,
            open(os.path.join(path, f"network_test_results-{time.time()}.json"), "w"),
        )


def get_unique_id() -> str:
    """Get a unique id based on the current time"""
    return str(time.time())
