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

from classes import Reinforcement

logger = logging.getLogger("root")
from disk import dump_agent, dump_results


def train_reinforcement_network(
    agent: object,
    input: Reinforcement.TrainingInput,
    args: argparse.Namespace,
    config: configparser.ConfigParser,
) -> Reinforcement.TrainingOutput:
    logger.info("Training reinforcement network...")

    output = Reinforcement.TrainingOutput(agent=agent, episodes=[])
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
        ) in enumerate(agent.train(input.num_episodes, args.render)):
            logger.info(f"Episode {episode}...")

            output.episodes.append(
                Reinforcement.Episode(
                    states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    losses=losses,
                    next_states=np.array(next_states),
                    dones=np.array(dones),
                    next_actions=np.array(next_action),
                )
            )

            if args.live_update:
                yield output.episodes[-1]

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

            dump_agent(config, agent)

            logger.info("Done dumping agent.")

        if args.save_results:
            logger.info("Saving results to disk...")

            if not os.path.exists(config["DEFAULT"]["results_dump_path"]):
                logger.debug(
                    "Results dump path does not exist, creating directory at path"
                )
                os.makedirs(config["DEFAULT"]["results_dump_path"])

            if output is None:
                raise Exception("Output is None, cannot save results to disk.")

            dump_results(config, agent.hashname, output)

            logger.info("Done saving results to disk.")

        logger.info("Done training reinforcement network.")

        return output
