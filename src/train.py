""" Training functions for networks and agents. """
import torch
import logging
import numpy as np
import argparse
import time
import os
import pickle
from typing import *
from dataclasses import dataclass
import configparser

logger = logging.getLogger("root")


@dataclass
class ReinforcementTrainingOutput:
    """Used to store the output of the reinforcement training process."""

    class Episode:
        """Contains information about a single episode."""

        losses: Any
        states: np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        next_states: np.ndarray
        dones: np.ndarray
        next_actions: np.ndarray

        def __init__(
            self, states, actions, rewards, losses, next_states, dones, next_actions
        ):
            self.losses = losses
            self.states = np.array(states)
            self.actions = np.array(actions)
            self.rewards = np.array(rewards)
            self.next_states = np.array(next_states)
            self.dones = np.array(dones)
            self.next_action = np.array(next_actions)

    agent: object
    episodes: List[Episode]


def train_reinforcement_network(
    agent: object,
    episodes: int,
    args: argparse.Namespace,
    config: configparser.ConfigParser,
) -> ReinforcementTrainingOutput:
    logger.info("Training reinforcement network...")

    output = ReinforcementTrainingOutput(agent=agent, episodes=[])
    try:
        # Agents are required to yield these values every episode, values are allowed to be empty in case we don't care about losses or any other field.
        for episode, (
            states,
            actions,
            rewards,
            losses,
            next_states,
            dones,
            next_action,
        ) in enumerate(agent.train(episodes, args.render)):
            logger.info(f"Episode {episode}...")

            output.episodes.append(
                ReinforcementTrainingOutput.Episode(
                    states=np.array(states) or [],
                    actions=np.array(actions) or [],
                    rewards=np.array(rewards) or [],
                    losses=losses or [],
                    next_states=np.array(next_states) or [],
                    dones=np.array(dones) or [],
                    next_actions=np.array(next_action) or [],
                )
            )
    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        raise e
    finally:
        logger.info("If '--dump' was specified, the agent will be dumped to disk")
        if args.dump:
            logger.info("Dumping agent...")

            if not os.path.exists(config["DEFAULT"]["agent_dump_path"]):
                logger.debug(
                    "Agent dump path does not exist, creating directory at path"
                )

                os.makedirs(config["DEFAULT"]["agent_dump_path"])

            with open(
                os.path.join(
                    config["DEFAULT"]["agent_dump_path"],
                    f"{str(time.time())}-{agent.name}-episode-{str(episode)}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(agent, f)
            logger.info("Done dumping agent.")

    if args.save_results:
        logger.info("Saving results to disk...")

        if not os.path.exists(config["DEFAULT"]["results_dump_path"]):
            logger.debug("Results dump path does not exist, creating directory at path")
            os.makedirs(config["DEFAULT"]["results_dump_path"])

        if output is None:
            raise Exception("Output is None, cannot save results to disk.")

        with open(
            os.path.join(
                config["DEFAULT"]["results_dump_path"],
                str(time.time()) + "-" + agent.name + ".pkl",
            ),
            "wb",
        ) as f:
            pickle.dump(output, f)

        logger.info("Done saving results to disk.")

    logger.info("Done training reinforcement network.")

    return output
