""" Used to create graphs and live updates of the agent/network's performance"""

import matplotlib.pyplot as plt
import os
import logging

from disk import dump_graph


def live_report_reinforcement_agent(
    generator, args, config, agent_hashname, moving_average_window=30
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
            plt.suptitle(f"Current Episode {enum}")
            if len(output.rewards) != 0 and None not in output.rewards:
                episode_sums.append(sum(output.rewards))
                plt.subplot(211)
                plt.plot(episode_sums)
                plt.xlim(0, enum + 10)
                plt.ylim(0, max(episode_sums) + 10)
                plt.xlabel("Episode")
                plt.ylabel("Reward")

                # draw moving average
                if len(episode_sums) > moving_average_window:
                    plt.plot(
                        [
                            sum(episode_sums[i : i + moving_average_window])
                            / moving_average_window
                            for i in range(len(episode_sums) - moving_average_window)
                        ],
                        color="red",
                    )

                    plt.legend(["Reward", "Moving Average"])

            # if its not empty and does not contain none
            if len(output.losses) != 0 and None not in output.losses:
                plt.subplot(212)
                plt.plot(output.losses)
                plt.xlim(0, enum + 10)
                plt.ylim(min(output.losses), max(output.losses))
                plt.xlabel("Episode")
                plt.ylabel("Loss")

                # Draw moving average
                if len(output.losses) > moving_average_window:
                    plt.plot(
                        [
                            sum(output.losses[i : i + moving_average_window])
                            / moving_average_window
                            for i in range(len(output.losses) - moving_average_window)
                        ],
                        color="red",
                    )

                    plt.legend(["Loss", "Moving Average"])

            plt.draw()
            plt.pause(0.00001)
            plt.clf()
    except KeyboardInterrupt:
        logging.info("Live update interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logging.info("Live update interrupted by exception")
        raise e
    finally:
        if args.save_figure:
            logging.info("Saving figure to disk...")
            dump_graph(config, agent_hashname, plt)
            logging.info("Done saving figure to disk")
