""" """

import argparse
import configparser

import log

logger = log.setup_custom_logger("root")
logger.debug("main message")

import sys
import os

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import train
import torch.nn as nn
import torch.optim as optim

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Kish: A PyTorch-based neural network testing framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add arguments to argparser, done in a separate file to keep main.py clean
    from arguments import add_arguments

    add_arguments(parser)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("config.ini")

    class TestAgent:
        def train(self, episodes, render, *args, **kwargs):
            for i in range(episodes):
                yield [], [], [], [], [], [],

    agent = TestAgent()

    out = train.train_reinforcement_network(
        agent,
        10,
        args,
        config,
    )

    print(out)
