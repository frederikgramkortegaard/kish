""" Create graphs and live updates of the agent/network's performance"""

import os
import sys
import logging
import matplotlib.pyplot as plt
from typing import Generator
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.classes import Reinforcement, Network
from src.log import setup_custom_logger

logger = setup_custom_logger(__name__)


def save_network_loss_graph(
    path: str,
    input: Network.TrainingOutput,
    custom_name: str = None,
    moving_average_window: bool = 30,
):
    """Create a graph showing the network's loss over time"""

    logger.info(f"Saving network loss graph")

    fig = plt.figure()

    average_losses = []
    ## Get the average loss per epoch
    for epoch in input.epochs:
        average_losses.append(sum(epoch.losses) / len(epoch.losses))

    plt.plot(average_losses)
    plt.ylim(min(average_losses) - 0.1, max(average_losses) + 0.1)

    # Check if it's a dir
    if not os.path.isdir(path):
        os.mkdir(path)

    if custom_name is not None:
        path = os.path.join(path, f"{custom_name}.png")
    else:
        path = os.path.join(path, f"network_loss-{time.time()}.png")

    plt.savefig(path)


def report_network_loss(
    input: Network.TrainingOutput,
    moving_average_window: bool = 30,
):
    """Create a graph showing the network's loss over time"""

    logger.info(f"Reporting network loss")

    fig = plt.figure()

    average_losses = []
    ## Get the average loss per epoch
    for epoch in input.epochs:
        average_losses.append(sum(epoch.losses) / len(epoch.losses))

    plt.plot(average_losses)
    plt.ylim(min(average_losses) - 0.1, max(average_losses) + 0.1)

    plt.show()


def report_reinforcement_agent(
    input: Reinforcement.TrainingOutput,
    moving_average_window: bool = 30,
):
    """Create a graph showing the agent's performance over time"""

    logger.info(f"Reporting reinforcement agent")

    f = plt.figure()
    axarr = f.add_subplot(1, 1, 1)  # here is where you add the subplot to f

    episode_sums = []
    try:
        for enum, episode in enumerate(input.episodes):
            episode_sums.append(sum(episode.rewards))

        plt.suptitle(f"Reinforcement Network Training")
        plt.plot(episode_sums)

        ## Setup
        plt.xlim(0, enum + 10)
        plt.ylim(0, max(episode_sums) + 10)
        plt.xlabel("Episode")
        plt.ylabel("Reward")

        # Moving Average
        if len(episode_sums) > moving_average_window:
            plt.plot(
                [
                    sum(episode_sums[i : i + moving_average_window])
                    / moving_average_window
                    for i in range(len(episode_sums) - moving_average_window)
                ],
                color="red",
            )

            ## We put this here because if we placed it before the loop we wouldn't get the "Moving Average" label
            plt.legend(["Reward", f"Moving Average ({moving_average_window} episodes)"])

        plt.show(block=True)
    except KeyboardInterrupt:
        logging.info("Live update interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logging.info("Live update interrupted by exception")
        raise e
    finally:
        plt.show()


def live_report_reinforcement_agent(
    generator: Generator[Reinforcement.Episode, None, Reinforcement.TrainingOutput],
    moving_average_window: bool = 30,
):
    """Create and live-update a graph using the ReinforcementTrainingOutput data"""

    logger.info(f"Live reporting reinforcement agent")

    plt.ion()

    episode_sums = []
    try:
        for enum, output in enumerate(generator):
            plt.clf()
            plt.suptitle(
                f"Reinforcement Network Training - Live Update\nCurrent Episode {enum} (0 indexed)"
            )
            episode_sums.append(sum(output.rewards))

            plt.plot(episode_sums)

            ## Setup
            plt.xlim(0, enum + 10)
            plt.ylim(0, max(episode_sums) + 10)
            plt.xlabel("Episode")
            plt.ylabel("Reward")

            # Moving Average
            if len(episode_sums) > moving_average_window:
                plt.plot(
                    [
                        sum(episode_sums[i : i + moving_average_window])
                        / moving_average_window
                        for i in range(len(episode_sums) - moving_average_window)
                    ],
                    color="red",
                )

                ## We put this here because if we placed it before the loop we wouldn't get the "Moving Average" label
                plt.legend(
                    ["Reward", f"Moving Average ({moving_average_window} episodes)"]
                )

            plt.draw()
            plt.pause(0.000001)
        plt.show(block=True)
    except KeyboardInterrupt:
        logging.info("Live update interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logging.info("Live update interrupted by exception")
        raise e
    finally:
        plt.show()


def save_reinforcement_agent_rewards_graph(
    path: str,
    input: Network.TrainingOutput,
    custom_name: str = None,
    moving_average_window: bool = 30,
):
    """Create a graph showing the network's loss over time"""

    logger.info(f"Saving reinforcement agent rewards graph")

    fig = plt.figure()
    plt.ylabel("Average Reward pr. episode")
    plt.xlabel("Episode")

    average_rewards = []
    for episode in input.episodes:

        average_reward = sum(episode.rewards) / len(episode.rewards)
        average_rewards.append(average_reward)

    plt.plot(average_rewards)
    plt.ylim(min(average_rewards) - 0.1, max(average_rewards) + 0.1)

    # Check if it's a dir
    if not os.path.isdir(path):
        os.mkdir(path)

    if custom_name is not None:
        path = os.path.join(path, f"{custom_name}.png")
    else:
        path = os.path.join(path, f"reinforcement_rewards-{time.time()}.png")

    plt.savefig(path)
