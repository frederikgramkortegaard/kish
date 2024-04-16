""" TBD """

import numpy as np
import torch.nn as nn
import os
import sys
import torch


# Class for storing transition memories
# also includes functions for batching, both randomly and according to index, which we need for nstep learning
class ReplayMemory:
    def __init__(self, capacity, batch_size, input):
        self.capacity = capacity
        self.mem_counter = 0
        self.mem_size = 0

        self.state_memory = torch.zeros((self.capacity, input), dtype=torch.float32).to(
            device
        )
        self.new_state_memory = torch.zeros(
            (self.capacity, input), dtype=torch.float32
        ).to(device)
        self.action_memory = torch.zeros(self.capacity, 17, dtype=torch.float32).to(
            device
        )
        self.next_action_memory = torch.zeros(
            self.capacity, 17, dtype=torch.float32
        ).to(device)
        self.reward_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.terminal_memory = torch.zeros(self.capacity, dtype=torch.float32).to(
            device
        )
        self.max_size = self.capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0

    # Stores a transition, while checking for the n-step value passed
    # if the n-step buffer is not larger than n, we only store the transition there
    def store_transition(self, state, action, reward, state_, next_action, done):

        state = torch.tensor(state).to(device)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        state_ = torch.tensor(state_).to(device)
        next_action = torch.tensor(next_action)

        if done == False:
            done = 0
        else:
            done = 1
        done = torch.tensor(done).to(device)

        self.state_memory[self.index] = state
        self.action_memory[self.index] = action
        self.new_state_memory[self.index] = state_
        self.reward_memory[self.index] = reward
        self.next_action_memory[self.index] = next_action
        self.terminal_memory[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size += 1

    # returns a random sample using indexes
    def sample_batch(self):

        indices = np.random.choice(self.capacity, size=self.batch_size, replace=False)

        return dict(
            states=self.state_memory[indices],
            actions=self.action_memory[indices],
            new_states=self.new_state_memory[indices],
            rewards=self.reward_memory[indices],
            next_actions=self.next_action_memory[indices],
            dones=self.terminal_memory[indices],
            indices=indices,
        )

    def __len__(self):
        return self.size
