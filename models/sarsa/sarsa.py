""" Implementation of a Deep SARSA Agent to solve the CartPole-v1 environment. """

import gym
import torch
import random
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from typing import Generator, List, Tuple


device = "cpu"


class Network(nn.Module):
    def __init__(self, input_size, output_size):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_size)

        # Initialize weights using orthogonal initialization
        nn.init.orthogonal_(self.fc1.weight)
        nn.init.orthogonal_(self.fc2.weight)
        nn.init.orthogonal_(self.fc3.weight)

    def forward(self, x) -> torch.Tensor:
        x = torch.tanh(self.fc1(x))
        x = torch.tanh(self.fc2(x))
        x = self.fc3(x)
        return x


class Memory:
    def __init__(self, size):
        self.size = size
        self.memory = []
        self.position = 0

    def push(self, state, action, reward, next_state, next_action, done) -> None:
        if len(self.memory) < self.size:
            self.memory.append(None)
        self.memory[self.position] = (
            state,
            action,
            reward,
            next_state,
            next_action,
            done,
        )
        self.position = (self.position + 1) % self.size

    def sample(
        self, batch_size
    ) -> List[Tuple[torch.Tensor, int, int, torch.Tensor, int, bool]]:
        return random.sample(self.memory, batch_size)


class Agent:
    """Deep SARSA (Replay Buffer Off-Policy Variation) Agent"""

    def __init__(
        self,
        env,
        n_inputs,
        n_outputs,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        batch_size,
        memory_size,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size

        self.model = Network(n_inputs, n_outputs)
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, weight_decay=0.001
        )
        self.memory = Memory(self.memory_size)

    def get_action(self, state) -> int:
        """Get an action from the agent
        using an epsilon-greedy policy.
        """

        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()

        state = torch.tensor(state, dtype=torch.float, device="cuda")
        action = self.model(state)
        return torch.argmax(action).item()

    def update(self) -> None:
        """Process a batch of experiences, calculate the q-values,
        and propagate the loss backwards through the network.
        """

        if len(self.memory.memory) < self.memory_size:
            return

        batch = self.memory.sample(self.batch_size)

        # Convert batch to tensors, this speeds up computation a lot
        batched_state = torch.tensor(
            np.array([b[0] for b in batch]), dtype=torch.float, device="cuda"
        )

        batched_action = torch.tensor(
            np.array([b[1] for b in batch]), dtype=torch.long, device="cuda"
        )
        batched_reward = torch.tensor(
            np.array([b[2] for b in batch]), dtype=torch.float, device="cuda"
        )
        batched_next_state = torch.tensor(
            np.array([b[3] for b in batch]), dtype=torch.float, device="cuda"
        )
        batched_next_action = torch.tensor(
            np.array([b[4] for b in batch]), dtype=torch.long, device="cuda"
        )

        batched_dones = torch.tensor(
            np.array([b[5] for b in batch]), dtype=torch.float, device="cuda"
        )

        q_value = (
            self.model(batched_state).gather(1, batched_action.unsqueeze(1)).squeeze(1)
        )
        next_q_value = (
            self.model(batched_next_state)
            .gather(1, batched_next_action.unsqueeze(1))
            .squeeze(1)
            .detach()
        )
        expected_q_value = (
            batched_reward + next_q_value * (1 - batched_dones) * self.gamma
        )

        loss = self.loss_fn(q_value, expected_q_value)
        self.optimizer.zero_grad()

        loss.backward()
        self.optimizer.step()

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
        return loss

    def train(self, episodes, render) -> Generator[int, None, None]:
        """Train the agent for a given number of episodes."""

        for episode in range(episodes):
            done = False
            truncated = False

            state = self.env.reset()
            action = self.get_action(state)
            episode_reward = 0

            states = []
            actions = []
            rewards = []
            losses = []
            next_states = []
            dones = []
            next_actions = []

            while not done or not truncated:
                if render:
                    self.env.render()

                next_state, reward, done, truncated = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                next_action = self.get_action(next_state)
                next_actions.append(next_action)
                self.memory.push(state, action, reward, next_state, next_action, done)

                loss = self.update()
                losses.append(loss)
                state = next_state
                action = next_action

                episode_reward += reward

            yield (
                states,
                actions,
                rewards,
                losses,
                next_states,
                dones,
                next_actions,
            )
