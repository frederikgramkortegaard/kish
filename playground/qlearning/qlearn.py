""" Implementation of a Deep Q-learning Agent to solve the Classic Control environments. """

import torch
import numpy as np
import torch.nn as nn
import os
import math
import sys
from collections import deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

import modules.Optimizer as optim

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initializes weights with orthonormal weights row-wise
def Orthonorm_weight(weight):
    ones = (
        torch.ones_like(weight).data.normal_(0, math.sqrt(2 / (weight.numel()))).t()
    )  # We choose Microsoft init as our choice of basis for the vector-space

    for i in range(1, int(ones.shape[0])):
        projection_neg_sum = torch.zeros_like(ones[i, :])
        for j in range(i):
            projection_neg_sum.data.add_(
                (
                    ((ones[i, :].t() @ ones[j, :]) / (ones[j, :].t() @ ones[j, :]))
                    * ones[j, :]
                )
            )
        ones[i, :].data.sub_(projection_neg_sum)

    ones /= torch.sqrt((ones**2).sum(-1, keepdim=True)) # We take out the norm from the coloumns, which will represent as the rows during forward passes

    return ones.t()  # Return Orthonormal basis

# Class for storing transition memories
# also includes functions for batching, both randomly and according to index, which we need for nstep learning
class ReplayMemory:
    def __init__(self, capacity, batch_size, input, n_step, gamma):
        self.gamma = gamma
        self.n_step = n_step
        self.capacity = capacity
        self.mem_counter = 0
        self.mem_size = 0

        self.state_memory = torch.zeros((self.capacity, input), dtype=torch.float32).to(
            device
        )
        self.new_state_memory = torch.zeros(
            (self.capacity, input), dtype=torch.float32
        ).to(device)
        self.action_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.next_action_memory = torch.zeros(self.capacity, dtype=torch.float32).to(
            device
        )
        self.reward_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.terminal_memory = torch.zeros(self.capacity, dtype=torch.float32).to(
            device
        )
        self.n_mem = deque(maxlen=n_step)
        self.max_size = self.capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0

    # Stores a transition, while checking for the n-step value passed
    # if the n-step buffer is not larger than n, we only store the transition there
    def store_transition(self, state, action, reward, state_, next_action, done):

        self.n_mem.append((state, action, reward, state_, next_action, done))
        if len(self.n_mem) < self.n_mem.maxlen:
            return

        # Calculate n_step reward
        temp = self.n_mem.copy()
        state, action, rew, _, _, _ = temp[0]
        sum_reward = 0
        for i in range(len(temp)):
            if i == 0:
                sum_reward += rew
                continue
            _, _, rew, state_, next_action, done = temp[i]
            if done:
                sum_reward += rew * self.gamma**i
                break
            sum_reward += rew * self.gamma**i
        reward = sum_reward
        
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

class Network(nn.Module):
    def __init__(self, inputs, hidden, actions):
        super(Network, self).__init__()
        self.input = inputs
        self.action = actions

        self.fc1 = nn.Linear(self.input, hidden)
        Orthonorm_weight(self.fc1.weight)

        self.action = nn.Linear(hidden, self.action)
        Orthonorm_weight(self.action.weight)

        self.layer_norm1 = nn.LayerNorm(hidden)

        self.f = nn.GELU()

    def forward(self, states):
        x = self.f(self.layer_norm1(self.fc1(states)))
        return self.action(x)


class Agent:

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
        n_step,
        hidden,
        tau
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.tau = tau
        self.n_step = n_step

        self.actor = Network(n_inputs, hidden, n_outputs).to(device)
        self.actor_target = Network(n_inputs, hidden, n_outputs).to(device)
        self.actor_target.load_state_dict(self.actor_target.state_dict())
        self.optimizer = optim.OrthAdam(self.actor.parameters(), lr=self.lr)
        self.memory = ReplayMemory(memory_size, self.batch_size, n_inputs, n_step=n_step, gamma=self.gamma)

    def update(self):

        samples = self.memory.sample_batch()
        state = samples["states"]
        action = samples["actions"].view(-1, 1)
        next_state = samples["new_states"]
        reward = samples["rewards"].view(-1, 1)
        next_action = samples["next_actions"].view(-1, 1)
        done = samples["dones"].view(-1, 1)

        mask = 1 - done

        Q_value = self.actor(state).gather(1, action.long())

        Q_next = torch.max(self.actor_target(next_state), dim=1, keepdim=True)[0].detach()

        Q_target = (reward + self.gamma**self.n_step * Q_next * mask).detach()

        loss = (Q_value - Q_target).pow(2).mean() # MSE loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for actor_weight, target_weight in zip(self.actor.parameters(), self.actor_target.parameters()):
            target_weight.data.copy_((1 - self.tau) * target_weight.data + self.tau * actor_weight.data)

        return loss.item()

    def train(self, timesteps, render):
        t = 0
        rewards = []
        losses = []

        while t <= timesteps:

            done = False
            truncated = False
            state, _ = self.env.reset()
            
            if np.random.uniform(size=1) >= self.epsilon and len(self.memory) >= self.batch_size:
                with torch.no_grad():
                    self.actor.eval()
                    action = self.actor(torch.FloatTensor(state).to(device).unsqueeze(0)).argmax(dim=-1).detach().cpu().item()
                    self.actor.train()
            else:
                action = self.env.action_space.sample()

            while not done and not truncated:

                next_state, reward, done, truncated, _ = self.env.step(action)

                if np.random.uniform(size=1) >= self.epsilon and len(self.memory) >= self.batch_size:
                    with torch.no_grad():
                        self.actor.eval()
                        next_action = self.actor(torch.FloatTensor(state).to(device).unsqueeze(0)).argmax(dim=-1).detach().cpu().item()
                        self.actor.train()
                else:
                    next_action = self.env.action_space.sample()
                    if len(self.memory) >= self.batch_size:
                        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

                rewards.append(reward)

                self.memory.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    state_=next_state,
                    next_action=next_action,
                    done=done,
                )

                if len(self.memory) >= self.batch_size:
                    loss = self.update()
                else:
                    loss = 1
                
                losses.append(loss)
                state = next_state
                action = next_action

                if (t % 100) == 0:
                    print("timestep: ", t)
                    print("Reward: ", sum(rewards))
                    print("Loss: ", sum(losses))
                    print("Average reward: ", sum(rewards) / len(rewards))
                    print("Average loss: ", sum(losses) / len(losses))
                    if self.epsilon != self.epsilon_min:
                        print("Current epsilon: ", self.epsilon)
                    print()

                if (t % 1000) == 0:
                    yield (rewards, losses)
                    rewards = []
                    losses = []
                t += 1
                if not (t <= timesteps):
                    break

    def test(self, timesteps, render):
        t = 0
        states = []
        next_actions = []
        self.actor.eval()
        while t <= timesteps:

            done = False
            state, _ = self.env.reset()
            truncated = False

            j = 0
            
            with torch.no_grad():
                action = self.actor(torch.FloatTensor(state).to(device).unsqueeze(0)).argmax(dim=-1).detach().cpu().item()

            while not done and not truncated:

                next_state, _, done, truncated, _ = self.env.step(action)

                with torch.no_grad():
                    next_action = self.actor(torch.FloatTensor(state).to(device).unsqueeze(0)).argmax(dim=-1).detach().cpu().item()

                state = next_state
                action = next_action
                states.append(state)
                next_actions.append(action)

                if (t % 1000) == 0:
                    yield (states, next_actions)
                    states = []
                    next_actions = []

                j += 1
                t += 1
                if not (t <= timesteps):
                    break
