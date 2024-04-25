""" 
"""

import torch.nn as nn
import torch.distributions as dist
import torch
import sys
import os
import numpy as np
import math
import cv2
import matplotlib.pyplot as plt

from skimage.util import crop
from skimage.color import rgb2gray
from collections import deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.Optimizer import RNAdamWP
from modules.Attentionsplit import AttentionSplit
from modules.VBlinear import VBLinear

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Class for storing transition memories
# also includes functions for batching
class ReplayMemory:
    def __init__(
        self,
        capacity,
        batch_size,
        frames,
        height,
        width,
        n_step,
        gamma,
        hidden_dim,
    ):
        self.capacity = capacity
        self.mem_counter = 0
        self.mem_size = 0
        self.gamma = gamma
        self.hidden_dim = hidden_dim
        self.frames = frames
        self.n_hidden = hidden_dim

        self.state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        ).to(device)
        self.new_state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        )
        self.action_memory = torch.zeros(self.capacity, dtype=torch.float32)
        self.reward_memory = torch.zeros(self.capacity, dtype=torch.float32)
        self.done_memory = torch.zeros(self.capacity, dtype=torch.int32)
        self.hidden_memmory = torch.zeros(
            self.capacity, frames, self.n_hidden, dtype=torch.float32
        ).to(device)
        self.batch_size = batch_size
        self.n_mem = deque(maxlen=n_step)
        self.index = 0
        self.size = 0

    # Stores a transition
    def store_transition(self, state, new_state, action, reward, done, hidden):

        hidden = hidden.view(self.frames, self.n_hidden)
        self.n_mem.append(reward)
        if len(self.n_mem) < self.n_mem.maxlen:
            return

        # Calculate n_step reward
        temp = self.n_mem.copy()
        temp.reverse()
        for rew in temp:
            reward = reward + self.gamma * rew

        state = state.to(device)
        new_state = new_state.to(device)

        done = torch.tensor(done).to(device)
        reward = torch.tensor(reward).to(device)

        self.state_memory[self.index] = state
        self.new_state_memory[self.index] = new_state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done
        self.hidden_memmory[self.index] = hidden

        if self.size < self.capacity - 1:
            self.size += 1
            self.index += 1
        else:
            self.index = (self.index + 1) % self.capacity

    # returns a sample using indexes
    def sample_batch(self):
        indices = np.random.choice(
            self.capacity - self.batch_size, size=self.batch_size, replace=False
        )
        return dict(
            states=self.state_memory[indices],
            new_states=self.new_state_memory[indices],
            actions=self.action_memory[indices],
            rewards=self.reward_memory[indices],
            dones=self.done_memory[indices],
            indices=indices,
            hiddens=self.hidden_memmory[indices],
        )

    def __len__(self):
        return self.size


# Main ANN class, contains visual, sequantial and action layer activations for Atari Environments
class Network(nn.Module):
    def __init__(self, actions):
        super(Network, self).__init__()
        self.actions = actions

        self.vision = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(
                in_channels=16, out_channels=32, kernel_size=7, padding=3, groups=8
            ),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(
                in_channels=32, out_channels=64, kernel_size=3, padding=1, groups=16
            ),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.MaxPool2d(4),
        )

        for module in self.vision:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n))  # Microsoft Init

        # Forward sequence
        self.encoder = AttentionSplit(256, 256, 4)

        self.action = VBLinear(256, self.actions)

        self.to(device)

    def forward(self, states, hidden=None):
        states = states.view(
            states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3]
        )
        # Assuming states has shape (batch_size, num_frames, num_channels, height, width)
        batch_size, num_frames, _, _, _ = states.shape

        # Reshape states to combine batch_size and num_frames dimensions
        states = states.view(-1, states.shape[2], states.shape[3], states.shape[4])
        x = self.vision(states)
        # Reshape back to (batch_size, num_frames, -1)
        x = x.view(batch_size, num_frames, -1)

        x, c = self.encoder(x, hidden)

        return self.action(x[:, -1, :]), c
import time
# Main Reinforcement Agent class
# Contains a training loop, which utilizes a DQL to provide training losses
# We use a decreasing epsilon / random action schedule to force the agent to new state spaces
class Agent:
    def __init__(
        self,
        frames,
        n_step,
        batch_size,
        mem_size,
        gamma,
        lr,
        weight_decay,
        n_outputs,
        hidden_dim,
        env,
        height,
        width,
        epsilon,
        epsilon_decay,
        epsilon_min,
    ):

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.frames = frames
        self.hidden_dim = hidden_dim
        self.tau = 0.005
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        self.actor = Network(n_outputs).to(device)
        self.optim1 = RNAdamWP(
            self.actor.parameters(),
            lr,
            weight_decay=weight_decay,
        )

        self.actor_target = Network(n_outputs).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.memory = ReplayMemory(
            capacity=mem_size,
            batch_size=batch_size,
            frames=frames,
            n_step=n_step,
            hidden_dim=self.hidden_dim,
            gamma=self.gamma,
            height=height,
            width=width,
        )

    # Defines the DQL update rule for the agent
    def update(self):

        samples = self.memory.sample_batch()
        state = samples["states"].to(device)
        action = samples["actions"].to(device).view(-1, 1)
        next_state = samples["new_states"].to(device)
        reward = samples["rewards"].view(-1, 1).to(device)
        done = samples["dones"].view(-1, 1).to(device)
        hiddens = samples["hiddens"].to(device)

        mask = 1 - done

        a, next_hidden = self.actor(state, hiddens)
        Q_vals = a.gather(1, action.long())

        an, _ = self.actor_target(next_state, next_hidden)
        Q_next = an.argmax(dim=1).detach().unsqueeze(-1)

        Q_target = reward + (self.gamma * Q_next * mask).detach()

        loss = (Q_vals - Q_target).pow(2).mean() + 0.1 * self.actor.action.KL() # MSE loss

        self.optim1.zero_grad()
        loss.backward()
        self.optim1.step()

        for eval_param, target_param in zip(
            self.actor.parameters(), self.actor_target.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + eval_param.data * self.tau
            )

        return loss

    # We use frame stacking to create sequential data for the agent to learn
    def stack_frames(self, stacked_frames, state, is_new_episode, max_frames):
        if is_new_episode:
            stacked_frames = deque(
                [
                    np.zeros((state.shape[0]), dtype=np.float64)
                    for _ in range(self.frames)
                ],
                maxlen=max_frames,
            )

            stacked_frames.extend([state] * max_frames)
            stacked_state = np.stack(stacked_frames, axis=0)

        else:
            stacked_frames.append(state)
            stacked_state = np.stack(stacked_frames, axis=0)
            h, w = stacked_state[0].shape
            for i in range(stacked_state.shape[0]):
                if i == 0:
                    continue
                indices = np.random.choice(a=h*w, size=int(h*w*0.1), replace=False)
                stacked_state[i-1].flat[indices] = 0
                stacked_state[i-1] = stacked_state[i-1].reshape(h, w)
                stacked_state[i] = np.maximum(stacked_state[i], stacked_state[i-1])
                #plt.imshow(stacked_state[i], cmap="gray")
                #plt.draw()
                #plt.pause(0.00000000000000000001)

        return torch.FloatTensor(stacked_state), stacked_frames

    # Main training loop.
    # We use a frameskip for both the action and update, to provide schochasticity in the actions as well as saving GPU ressources
    def train(self, timesteps, render):
        t = 0
        rewards = []
        observed_rewards = []
        losses = []
        while t <= timesteps:

            done = False
            truncated = False
            state, info = self.env.reset()
            state = cv2.resize(
                rgb2gray(crop(state, ((13, 13), (15, 25), (0, 0)))), (84, 84)
            )

            stacked_frames = deque(
                [
                    np.zeros((state.shape[0]), dtype=np.float64)
                    for _ in range(self.frames)
                ],
                maxlen=self.frames,
            )
            state, stacked_frames = self.stack_frames(
                stacked_frames, state, True, self.frames
            )

            j = 0
            lives = info["lives"]

            hiddens = torch.zeros((self.frames, self.hidden_dim)).to(device)
            prev_hiddens = hiddens.clone()
            while not done and not truncated:


                if np.random.uniform(size=1) >= self.epsilon:
                    with torch.no_grad():
                        self.actor.eval()
                        action, hiddens = self.actor(
                            torch.FloatTensor(state).to(device).unsqueeze(0), hiddens
                        )
                        action = action.argmax(dim=-1).detach().cpu().item()
                        self.actor.train()
                else:
                    action = self.env.action_space.sample()
                    hiddens += torch.randn(hiddens.numel(), device=device).view(hiddens.shape) * state.std()


                next_state, reward, done, truncated, info = self.env.step(action)
                
                next_state, stacked_frames = self.stack_frames(
                    stacked_frames,
                    cv2.resize(
                        rgb2gray(crop(next_state, ((13, 13), (15, 25), (0, 0)))),
                        (84, 84),
                    ),
                    False,
                    self.frames,
                )

                rewards.append(reward)
    
                if reward > 0:
                    reward = min(reward, 5)
                if lives > info["lives"]:
                    lives = info["lives"]
                    reward -= 1

                observed_rewards.append(reward)
                self.memory.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    new_state=next_state,
                    done=done,
                    hidden=prev_hiddens,
                )
                prev_hiddens = hiddens

                if len(self.memory) >= self.memory.capacity - 1:
                    loss = self.update().item()
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                else:
                    loss = 1
                losses.append(loss)
                state = next_state

                if (t % 100) == 0:
                    print("timestep: ", t)
                    print("Reward: ", sum(rewards))
                    print("Observed reward: ", sum(observed_rewards))
                    print("Loss: ", sum(losses))
                    print("Average reward: ", sum(rewards) / len(rewards))
                    print("Average observed reward: ", sum(observed_rewards) / len(observed_rewards))
                    print("Average loss: ", sum(losses) / len(losses))
                    if self.epsilon != self.epsilon_min:
                        print("Current epsilon: ", self.epsilon)
                    print()
                if (t % 1000) == 0:
                    yield (rewards, losses)
                    rewards = []
                    observed_rewards = []
                    losses = []

                j += 1
                t += 1
