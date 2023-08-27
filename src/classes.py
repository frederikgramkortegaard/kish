""" """


from dataclasses import dataclass
import numpy as np
from typing import Any, List


@dataclass
class Reinforcement:
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

    class TrainingOutput:
        agent: object
        episodes: List["Episode"]

        def __init__(self, agent, episodes):
            self.agent = agent
            self.episodes = episodes

    class TrainingInput:
        num_episodes: int
        render: bool

        def __init__(self, num_episodes, render):
            self.num_episodes = num_episodes
            self.render = render
