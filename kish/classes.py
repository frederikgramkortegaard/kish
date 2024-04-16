""" Containers and dataclasses used throughout the project.
"""

import numpy as np
from dataclasses import dataclass
from typing import Any, List


@dataclass
class Network:
    """Used to store the output of the reinforcement training process."""

    class Epoch:
        """Contains information about a single epoch."""

        losses: Any

        def __init__(self, losses):
            self.losses = losses

    class TrainingOutput:
        epochs: List["Epoch"]

        def __init__(self, epochs):
            self.epochs = epochs

    class TrainingInput:
        trainloader: Any
        epochs: int
        device: str
        criterion: Any
        optimizer: Any
        scheduler: Any

        def __init__(
            self,
            trainloader,
            epochs,
            device,
            criterion,
            optimizer,
            scheduler,
        ):
            self.trainloader = trainloader
            self.epochs = epochs
            self.device = device
            self.criterion = criterion
            self.optimizer = optimizer
            self.scheduler = scheduler

    class TestingOutput:

        accuracy: int

        def __init__(self, accuracy):
            self.accuracy = accuracy

    class TestingInput:
        testloader: Any
        device: str

        def __init__(
            self,
            testloader,
            device,
        ):
            self.testloader = testloader
            self.device = device


@dataclass
class Reinforcement:
    """Used to store the output of the reinforcement training process."""

    class timesteps:
        """Contains information about a single episode."""

        losses: Any
        rewards: np.ndarray

        def __init__(
            self,
            rewards,
            losses,
        ):
            self.losses = losses
            self.rewards = np.array(rewards)

    class TrainingOutput:
        """Returned from the reinforcement training processes."""

        agent: object
        timesteps: List["timesteps"]

        def __init__(self, agent, timesteps):
            self.agent = agent
            self.timesteps = timesteps

    class TrainingInput:
        """Configuration for the reinforcement training processes."""

        num_timesteps: int
        render: bool  # @TODO : would be _great_ to move this into live_reporting but that's gonna take a little bit of work

        def __init__(self, num_timesteps, render: bool = False):
            self.num_timesteps = num_timesteps
            self.render = render

            assert self.num_timesteps > 0

    class RuntimeInput:
        pass

    class RuntimeOutput:
        pass
