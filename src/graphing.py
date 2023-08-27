""" Create graphs and live updates of the agent/network's performance"""

import os
import logging
import matplotlib.pyplot as plt

from typing import *
from classes import Reinforcement


def live_report_reinforcement_agent(
    generator: Generator[Reinforcement.Episode, None, Reinforcement.TrainingOutput],
    config: object,
    moving_average_window: bool = 30,
):
    """Create and live-update a graph using the ReinforcementTrainingOutput data"""

    if not os.path.exists(config["DEFAULT"]["path_to_graphs"]):
        logging.debug("Creating graphs directory")
        os.mkdir(config["DEFAULT"]["path_to_graphs"])

    plt.ion()
    plt.title("Reinforcement Network Training - Live Update")

    episode_sums = []
    try:
        for enum, output in enumerate(generator):
            plt.clf()
            plt.suptitle(f"Current Episode {enum}")
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
            plt.pause(0.00001)
        plt.show(block=True)
    except KeyboardInterrupt:
        logging.info("Live update interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logging.info("Live update interrupted by exception")
        raise e
    finally:
        plt.show()
        pass
