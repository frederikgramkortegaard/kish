""" Implementation of a Deep SARSA Agent to solve the CartPole-v1 environment. """

import torch
import numpy as np
import torch.nn as nn
import os 
import math
import sys

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.Optimizer import SGDNorm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Class for storing transition memories
# also includes functions for batching, both randomly and according to index, which we need for nstep learning
class ReplayMemory:
    def __init__(self, capacity, batch_size, input):
        self.capacity = capacity
        self.mem_counter = 0
        self.mem_size = 0

        self.state_memory = torch.zeros((self.capacity, input),
                                     dtype=torch.float32).to(device)
        self.new_state_memory = torch.zeros((self.capacity, input),
                                         dtype=torch.float32).to(device)
        self.action_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.next_action_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.reward_memory = torch.zeros(self.capacity,  dtype=torch.float32).to(device)
        self.terminal_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
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
        self.size +=1

    # returns a random sample using indexes
    def sample_batch(self):

        indices = np.random.choice(self.capacity, size=self.batch_size, replace=False)

        return dict ( states=self.state_memory[indices], actions=self.action_memory[indices], new_states=self.new_state_memory[indices],
                      rewards = self.reward_memory[indices], next_actions=self.next_action_memory[indices], dones=self.terminal_memory[indices], indices = indices, )

    def __len__(self):
        return self.size
    
class func(nn.Module):
    def __init__(self):
        super(func, self).__init__()

    def forward(self, input):
        
        return torch.where(input <= 1, torch.exp(input)-1, torch.log(input) + 1 + (1/math.sqrt(2)))

class Network(nn.Module):
    def __init__(self, inputs, actions):
        super(Network, self).__init__()
        self.input = inputs
        self.action = actions

        self.fc1 = nn.Linear(self.input, 256)
        nn.init.normal_(self.fc1.weight, 0, 1/self.fc1.weight.numel())
        nn.utils.parametrizations.weight_norm(self.fc1)

        self.action = nn.Linear(256, self.action)
        nn.init.normal_(self.action.weight, 0, 1/self.action.weight.numel())
        nn.utils.parametrizations.weight_norm(self.action)

        self.layer_norm1 = nn.LayerNorm(256)

        self.SELU = func()

    def forward(self, states):
        x = self.SELU(self.layer_norm1(self.fc1(states)))
        return self.action(x)

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

        self.model = Network(n_inputs, n_outputs).to(device)
        self.loss_fn = nn.SmoothL1Loss()
        self.optimizer = SGDNorm(self.model.parameters(), lr=0.1, weight_decay=1e-4)
        self.memory = ReplayMemory(memory_size, 256, n_inputs)

    def update(self):

        samples = self.memory.sample_batch()
        state = samples["states"]
        action = samples["actions"].view(-1,1)
        next_state = samples["new_states"]
        reward = samples["rewards"].view(-1,1)
        next_action = samples["next_actions"].view(-1,1)
        done = samples["dones"].view(-1, 1)

        mask = 1 - done
        
        Q_value = self.model(state).gather(1, action.long())

        Q_next = self.model(next_state).gather(1, next_action.long()).detach()

        Q_target = reward + (self.gamma * Q_next * mask).detach()

        loss = (Q_value - Q_target).pow(2).mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

    def train(self, episodes, render):
        for e in range(episodes):
            done = False
            truncated = False

            state = self.env.reset()
            if (np.random.uniform(0, 1) >= self.epsilon):
                with torch.no_grad():
                        self.model.eval()
                        action = torch.argmax(self.model(torch.FloatTensor(state).to(device))).item()
                        self.model.train()
            else:
                action = self.env.action_space.sample()
            episode_reward = 0

            states = []
            actions = []
            rewards = []
            losses = []
            next_states = []
            dones = []
            next_actions = []

            while not done or not truncated:
                if render and len(self.memory) >= self.memory.capacity -1:
                    self.env.render()

                next_state, reward, done, truncated = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                if (np.random.uniform(0, 1) >= self.epsilon):
                    with torch.no_grad():
                        self.model.eval()
                        next_action = torch.argmax(self.model(torch.FloatTensor(next_state).to(device))).item()
                        self.model.train()
                else:
                    next_action = self.env.action_space.sample()
                next_actions.append(next_action)
                self.memory.store_transition(state, action, reward, next_state, next_action, done)

                if len(self.memory) >= self.memory.capacity -1:
                    loss = self.update()
                    self.epsilon = max(self.epsilon * 0.99, 0.01)
                else:
                    loss = 1
                losses.append(loss)
                state = next_state
                action = next_action

                episode_reward += reward

            print(e)
            print(episode_reward)
            yield (
                states,
                actions,
                rewards,
                losses,
                next_states,
                dones,
                next_actions,
            )


    def test(self, episodes, render):
         for _ in range(episodes):
            done = False
            truncated = False

            state = self.env.reset()
            if (np.random.uniform(0, 1) >= self.epsilon):
                with torch.no_grad():
                        self.model.eval()
                        action = torch.argmax(self.model(torch.FloatTensor(state).to(device))).item()
                        self.model.train()
            else:
                action = self.env.action_space.sample()
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

                if (np.random.uniform(0, 1) >= self.epsilon):
                    with torch.no_grad():
                        self.model.eval()
                        next_action = torch.argmax(self.model(torch.FloatTensor(next_state).to(device))).item()
                        self.model.train()
                else:
                    next_action = self.env.action_space.sample()
                next_actions.append(next_action)
                state = next_state
                action = next_action

                episode_reward += reward
            
            print(episode_reward)
            yield (
                states,
                actions,
                rewards,
                losses,
                next_states,
                dones,
                next_actions,
            )


