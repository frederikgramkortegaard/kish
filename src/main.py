""" Kish: A PyTorch-based neural network testing & Evaluation framework """
# Sys settrace
import argparse
import configparser

import log

logger = log.setup_custom_logger("root")
logger.debug("main message")

import sys
import os
import gym
import time

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent import *
from train import train_reinforcement_network
from graphing import live_report_reinforcement_agent
from versioning import verify_agent, name_agent
from classes import Reinforcement

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

    verify_agent(agent)
    name_agent(agent)
    print(agent.hashname)

    # Create input to train_reinforcement_network
    input = Reinforcement.TrainingInput(
        num_episodes=100,
        render=False,
    )

    live_report_reinforcement_agent(
        train_reinforcement_network(agent=agent, input=input, args=args, config=config),
        args,
        config,
        agent.hashname,
    )
