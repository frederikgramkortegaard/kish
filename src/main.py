""" Kish: A PyTorch-based neural network testing & Evaluation framework """

import argparse
import configparser

import log

logger = log.setup_custom_logger("root")
logger.debug("main message")

import sys
import os
import gym

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import *
import train

if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Kish: A PyTorch-based neural network testing & Evaluation framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Add arguments to argparser, done in a separate file to keep main.py clean
    from arguments import add_arguments

    add_arguments(parser)

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read("config.ini")

    agent = Agent(
        env=gym.make("CartPole-v1"),
        n_inputs=4,
        n_outputs=2,
        lr=0.01,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=10000,
    )

    output = train.train_reinforcement_network(
        agent=agent, episodes=1, args=args, config=config
    )
