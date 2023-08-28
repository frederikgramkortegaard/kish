""" Kish: A PyTorch-based neural network testing & Evaluation framework """
import os
import sys
import log
import gym
import time
import pickle
import hashlib
import argparse
import configparser

logger = log.setup_custom_logger("root")
logger.debug("main message")

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents import DeepSARSA
from classes import Reinforcement
from graphing import live_report_reinforcement_agent, report_reinforcement_agent
from train import train_reinforcement_agent, iterative_train_reinforcement_agent


if __name__ == "__main__":
    # Setup argument parser
    parser = argparse.ArgumentParser(
        description="Kish: A PyTorch-based neural network testing & Evaluation framework",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "-c",
        "--config",
        dest="config_file",
        default="config.ini",
        type=str,
        help="Configuration file to use",
    )

    parser.add_argument(
        "-r",
        "--render",
        dest="render",
        action="store_true",
        help="Render the environment for Reinforcement Learning Training",
        default=False,
    )

    args = parser.parse_args()

    config = configparser.ConfigParser()
    config.read(args.config_file)

    agent = DeepSARSA(
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

    #### == STATIC == ####

    # Statically train a model and get the episodic outputs
    output = train_reinforcement_agent(
        agent=agent,
        input=Reinforcement.TrainingInput(num_episodes=50),
    )

    # Generate a graph from the output
    report_reinforcement_agent(
        input=output,
    )

    #### == LIVE == ####

    # Live update a graph from the iteratively generated output
    output_generator = iterative_train_reinforcement_agent(
        agent=agent,
        input=Reinforcement.TrainingInput(num_episodes=500, render=args.render),
    )

    # Iterate over the generator and generate a graph
    live_report_reinforcement_agent(
        generator=output_generator,
    )
