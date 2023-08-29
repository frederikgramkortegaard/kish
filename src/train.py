""" Training functions for networks and agents. 


@TODO : split reinforcement and normal network logic in separate files"""
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


def train_reinforcement_agent(
    agent: object, input: Reinforcement.TrainingInput
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
        ) in enumerate(agent.train(input.num_episodes, render=False)):
            logging.info(f"Episode {episode}...")
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

    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        raise e
    finally:
        logger.info("Done training reinforcement network.")

        return output


def iterative_train_reinforcement_agent(
    agent: object,
    input: Reinforcement.TrainingInput,
) -> Generator[Reinforcement.Episode, None, Reinforcement.TrainingOutput]:
    logger.info("Iteratively training reinforcement network...")

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
        ) in enumerate(agent.train(input.num_episodes, input.render)):
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

            yield output.episodes[-1]

    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        raise e
    finally:
        logger.info("Done training reinforcement network.")

        return output


def run_reinforcement_agent(
    agent: object, input: Reinforcement.RuntimeInput
) -> Reinforcement.RuntimeOutput:
    logger.info("Running reinforcement network...")
    pass


def iterative_run_reinforcement_agent(
    agent: object,
    input: Reinforcement.RuntimeInput,
) -> Generator[Reinforcement.Episode, None, Reinforcement.RuntimeOutput]:
    logger.info("Iteratively running reinforcement network...")
    pass
