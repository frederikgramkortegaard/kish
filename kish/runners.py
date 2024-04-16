""" Training functions for networks and agents.  """

import torch
import logging
import numpy as np
from typing import *
from dataclasses import dataclass

from kish.classes import Network
from kish.classes import Reinforcement
from kish.log import setup_custom_logger

logger = setup_custom_logger(__name__)


def test_network(
    network: object,
    input: Network.TestingInput,
) -> Network.TestingOutput:
    """Test a network on a dataloader"""

    logger.info("Testing network...")

    with torch.no_grad():
        network.eval()
        correct = 0
        total = 0
        for enum, (inputs, labels) in enumerate(input.testloader):
            outputs = network(inputs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            if enum >= 4:
                break

    logger.info("Done testing network.")

    return Network.TestingOutput(
        accuracy=100 * correct / total,
    )


def train_network(
    network: object,
    input: Network.TrainingInput,
) -> Network.TrainingOutput:
    """Train a network on a dataloader"""

    logger.info("Training network...")
    output = Network.TrainingOutput(epochs=[])

    try:
        for epoch in range(input.epochs):
            logger.info(f"Epoch {epoch}...")
            network.train()

            output.epochs.append(Network.Epoch(losses=[]))

            for i, data in enumerate(input.trainloader):
                inputs, labels = data
                input.optimizer.zero_grad()

                outputs = network(inputs)
                loss = input.criterion(outputs, labels)
                epoch_loss = loss.item()
                output.epochs[-1].losses.append(epoch_loss)
                loss.backward()
                input.optimizer.step()

                if i >= 1:  # @TODO : REMOVE WHEN DONE TESTING GRAPHIN
                    break

            input.scheduler.step()
            if epoch >= 4:
                break
    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        logger.error(e)

        raise e
    finally:
        logger.info("Done training network.")
        return output


def train_reinforcement_agent(
    agent: object, input: Reinforcement.TrainingInput
) -> Reinforcement.TrainingOutput:
    """Train a reinforcement agent on a gym environment"""

    logger.info("Training reinforcement network...")
    output = Reinforcement.TrainingOutput(agent=agent, episodes=[])

    assert input.num_episodes > 0, "Number of episodes must be greater than 0"
    assert input.render == False, "Rendering is only supported for iterative training"

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
        logger.error(e)

        raise e
    finally:
        logger.info("Done training reinforcement network.")

        return output


def iterative_train_reinforcement_agent(
    agent: object,
    input: Reinforcement.TrainingInput,
) -> Generator[Reinforcement.timesteps, None, Reinforcement.TrainingOutput]:
    """Train a reinforcement agent on a gym environment, this
    function is a generator that yields episodes as they are completed.
    """

    logger.info("Iteratively training reinforcement network...")

    output = Reinforcement.TrainingOutput(agent=agent, timesteps=[])
    try:
        # Agents are required to yield these values every episode, values are allowed to be empty in case we don't care about losses or any other field.
        for timesteps, (
            rewards,
            losses,
        ) in enumerate(agent.train(timesteps=input.num_timesteps, render=input.render)):
            logger.info(f"1000 timesteps {timesteps}...")

            output.timesteps.append(
                Reinforcement.timesteps(
                    rewards=np.array(rewards),
                    losses=losses,
                )
            )

            yield output.timesteps[-1]

    except KeyboardInterrupt:
        logger.debug("Training interrupted by user")
        raise KeyboardInterrupt
    except Exception as e:
        logger.debug("Training interrupted by exception")
        logger.error(e)
        raise e
    finally:
        logger.info("Done training reinforcement network.")

        return output


def run_reinforcement_agent(
    agent: object, input: Reinforcement.RuntimeInput
) -> Reinforcement.RuntimeOutput:
    logger.info("Running reinforcement network...")
    raise NotImplementedError


def iterative_run_reinforcement_agent(
    agent: object,
    input: Reinforcement.RuntimeInput,
) -> Generator[Reinforcement.timesteps, None, Reinforcement.RuntimeOutput]:
    logger.info("Iteratively running reinforcement network...")
    raise NotImplementedError
