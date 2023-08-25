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


def train_network(
    network: torch.nn.Module,
    X: List[Any],
    y: List[int],
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    epochs: int,
) -> torch.nn.Module:
    """Train a network on the given dataset (X, y) and return the trained network. Can also be used to fine-tune a network."""

    logger.debug("Training network...")

    for epoch in range(epochs):
        logger.debug(f"Epoch {epoch}...")
        for i, x in enumerate(X):
            y_pred = network(x)
            loss = loss_fn(y_pred, y[i])
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

    logger.debug("Done training network.")

    return network


@dataclass
class ReinforcementTrainingOutput:
    """Used to store the output of the reinforcement training process."""

    class Episode:
        """Contains information about a single episode."""

        losses: np.ndarray
        states: np.ndarray
        actions: np.ndarray
        rewards: np.ndarray
        next_states: np.ndarray
        dones: np.ndarray

        def __init__(self, states, actions, rewards, losses, next_states, dones):
            self.losses = np.array(losses)
            self.states = np.array(states)
            self.actions = np.array(actions)
            self.rewards = np.array(rewards)
            self.next_states = np.array(next_states)
            self.dones = np.array(dones)

    agent: object
    episodes: List[Episode]


def train_reinforcement_network(
    agent: object,
    episodes: int,
    args: argparse.Namespace,
    config: configparser.ConfigParser,
) -> ReinforcementTrainingOutput:
    logger.debug("Training reinforcement network...")

    output = ReinforcementTrainingOutput(agent=agent, episodes=[])
    try:
        for episode, (
            states,
            actions,
            rewards,
            losses,
            next_states,
            dones,
        ) in enumerate(agent.train(episodes, args.render)):
            logger.debug(f"Episode {episode}...")

            output.episodes.append(
                ReinforcementTrainingOutput.Episode(
                    states=np.array(states),
                    actions=np.array(actions),
                    rewards=np.array(rewards),
                    losses=np.array(losses),
                    next_states=np.array(next_states),
                    dones=np.array(dones),
                )
            )
    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        raise e
    finally:
        logger.debug("If '--dump' was specified, the agent will be dumped to disk")
        if args.dump:
            logger.debug("Dumping agent...")
            with open(
                os.path.join(config["DEFAULT"]["agent_dump_path"], str(time.now())),
                "wb",
            ) as f:
                pickle.dump(agent, f)
            logger.debug("Done dumping agent.")

    logger.debug("Done training reinforcement network.")

    return output
