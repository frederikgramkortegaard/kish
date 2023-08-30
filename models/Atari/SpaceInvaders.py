import torch
import torch.nn as nn
import numpy as np
import gym
from skimage.color import rgb2gray
from collections import deque
import cv2
import math
from skimage.util import crop
import torchvision.transforms as transforms
import sys
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.Atari.Optimizer import OrthAdam

device = torch.device("cuda")


# Class for storing transition memories
# also includes functions for batching
class ReplayMemory:
    def __init__(self, capacity, batch_size, frames, height, width, n_step, gamma):
        self.capacity = capacity
        self.mem_counter = 0
        self.mem_size = 0
        self.gamma = gamma

        self.state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        ).to(device)
        self.new_state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        )
        self.action_memory = torch.zeros(self.capacity, dtype=torch.int32)
        self.reward_memory = torch.zeros(self.capacity, dtype=torch.float32)
        self.done_memory = torch.zeros(self.capacity, dtype=torch.int32)
        self.batch_size = batch_size
        self.n_mem = deque(maxlen=n_step)
        self.index = 0
        self.size = 0

    # Stores a transition
    def store_transition(self, state, new_state, action, reward, done):
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
        if isinstance(action, torch.Tensor):
            action = action.to(device)
        else:
            action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        done = torch.tensor(done).to(device)

        self.state_memory[self.index] = state
        self.new_state_memory[self.index] = new_state
        self.action_memory[self.index] = action
        self.reward_memory[self.index] = reward
        self.done_memory[self.index] = done

        self.index = (self.index + 1) % self.capacity
        self.size += 1

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
        )

    def __len__(self):
        return self.size


class f(nn.Module):
    def __init__(self):
        super(f, self).__init__()

    def forward(self, x):
        return (
            torch.sinh(nn.functional.sigmoid(x)) * x
        )  # Generalized Gaussian Error unit


class Network(nn.Module):
    def __init__(self, inputs, actions):
        super(Network, self).__init__()
        self.input = inputs
        self.actions = actions

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding=5)
        n = self.conv1.kernel_size[0] * self.conv1.kernel_size[1] * self.conv1.out_channels
        nn.init.normal_(self.conv1.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3)
        n = self.conv2.kernel_size[0] * self.conv2.kernel_size[1] * self.conv2.out_channels
        nn.init.normal_(self.conv2.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        n = self.conv3.kernel_size[0] * self.conv3.kernel_size[1] * self.conv3.out_channels
        nn.init.normal_(self.conv3.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init
        self.bn3 = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        n = self.conv4.kernel_size[0] * self.conv4.kernel_size[1] * self.conv4.out_channels
        nn.init.normal_(self.conv1.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init
        self.bn4 = nn.BatchNorm2d(128)

        self.maxpool = nn.MaxPool2d(3)
        self.avgpool = nn.AvgPool2d(3)

        self.dropout = nn.Dropout(0.5)
        self.action = nn.Sequential(
            nn.Linear(384, 384),
            nn.LayerNorm(384),
            f(),
            nn.Dropout(0.5),
            nn.Linear(384, self.actions),
        )
        for module in self.action:
            if isinstance(module, nn.LayerNorm):
                nn.init.normal_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)

        self.f = f()

        self.to(device)

    def forward(self, states):
        states = states.view(
            states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3]
        )

        xs = []
        i = 0
        while i != states.shape[1]:
            state = states[:, i, :, :]
            x = self.dropout(self.maxpool(self.f(self.bn1(self.conv1(state)))))
            x = self.dropout(self.maxpool(self.f(self.bn2(self.conv2(x)))))
            x = self.dropout(self.maxpool(self.f(self.bn3(self.conv3(x)))))
            x = self.dropout(self.avgpool(self.f(self.bn4(self.conv4(x)))))
            xs.append(x.flatten(1))
            i += 1
        x = torch.stack(xs, dim=1).flatten(1)

        return self.action(x)


class Agent:
    def __init__(self, frames, n_step, batch_size, mem_size):
        self.game = gym.make("SpaceInvaders-v0")
        state = cv2.resize(
            rgb2gray(crop(self.game.reset(), ((13, 13), (15, 25), (0, 0)))), (84, 84)
        )
        self.steps = 0
        self.frames = frames
        self.game.seed(0)
        self.eval = Network(state.flatten().shape[0], self.game.action_space.n).to(
            device
        )
        self.eval_target = Network(
            state.flatten().shape[0], self.game.action_space.n
        ).to(device)
        self.eval_target.load_state_dict(self.eval.state_dict())

        self.memory = ReplayMemory(
            mem_size, batch_size, self.frames, state.shape[0], state.shape[1], n_step, 0.99
        )
        self.optimizer = OrthAdam(self.eval.parameters(), lr=0.00003, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 100, 2
        )
        self.transform = transforms.Compose(
            [
                transforms.GaussianBlur([11, 11], (0.1, 3)),
                transforms.RandomErasing(),
                transforms.RandomRotation(degrees=[-180, 180]),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        self.gamma = 0.99
        self.TAU = 0.0005
        self.epsilon = 0.99

    def update(self):
        self.eval.train()
        self.eval_target.train()

        samples = self.memory.sample_batch()

        state = self.transform(samples["states"].squeeze(1))
        new_state = self.transform(samples["new_states"].squeeze(1)).to(device)
        action = samples["actions"].view(-1, 1).to(device)
        reward = samples["rewards"].view(-1, 1).to(device)
        done = samples["dones"].view(-1, 1).to(device)

        mask = 1 - done

        Q_value = self.eval(state).gather(1, action.long())

        Q_next = torch.mean(self.eval_target(new_state), dim=-1).view(-1, 1).detach()

        Q_target = (reward + (self.gamma * Q_next * mask)).detach()

        loss = ((Q_value - Q_target) ** 2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for eval_param, target_param in zip(
            self.eval.parameters(), self.eval_target.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.TAU) + eval_param.data * self.TAU
            )

        self.steps += 1

        return loss

    def stack_frames(self, stacked_frames, state, is_new_episode, max_frames):
        if is_new_episode:
            stacked_frames = deque(
                [
                    np.zeros((1, state.shape[0], state.shape[1]), dtype=np.float64)
                    for _ in range(self.frames)
                ],
                maxlen=max_frames,
            )

            stacked_frames.extend([state] * max_frames)
            stacked_state = np.stack(stacked_frames, axis=0)

        else:
            stacked_frames.append(state)
            stacked_state = np.stack(stacked_frames, axis=0)

        return torch.FloatTensor(stacked_state), stacked_frames

    def train(self, episodes, render):

        for _ in range(episodes):

            state = cv2.resize(
                rgb2gray(crop(self.game.reset(), ((13, 13), (15, 25), (0, 0)))),
                (84, 84),
            )

            stacked_frames = deque(
                [
                    np.zeros((1, state.shape[0], state.shape[1]), dtype=np.float64)
                    for _ in range(self.frames)
                ],
                maxlen=self.frames,
            )
            state, stacked_frames = self.stack_frames(
                stacked_frames, state, True, self.frames
            )

            done = False
            truncated = False

            self.lives = self.game.ale.lives()
            j = 0
            rewards = []
            losses = []
            while not done or not truncated:
                current_loss = 0
                if len(self.memory) > (self.memory.capacity) and render == True:
                    self.game.render()

                if j % 3 == 0:
                    if (np.random.uniform(0, 1) >= self.epsilon) and len(
                        self.memory
                    ) > (self.memory.capacity):
                        with torch.no_grad():
                            self.eval.eval()
                            action_choice = self.eval(
                                torch.FloatTensor(state).unsqueeze_(0).to(device)
                            )
                            action = torch.argmax(action_choice).item()
                    else:
                        action_choice = np.random.normal(size=self.game.action_space.n)
                        action = np.argmax(action_choice)

                next_state, reward, done, truncated = self.game.step(action)
                true_reward = reward

                reward = min(1, reward)

                rewards.append(true_reward)

                if self.lives > self.game.ale.lives():
                    reward -= 1
                    self.lives = self.game.ale.lives()

                next_state, stacked_frames = self.stack_frames(
                    stacked_frames,
                    cv2.resize(
                        rgb2gray(crop(next_state, ((13, 13), (15, 25), (0, 0)))),
                        (84, 84),
                    ),
                    False,
                    self.frames,
                )

                self.memory.store_transition(state, next_state, action, reward, done)

                if len(self.memory) > (self.memory.capacity) and j % 3 == 0:
                    current_loss = self.update().item()
                    self.epsilon = max(self.epsilon * 0.999995, 0.01)

                losses.append(current_loss)
                state = next_state

                j += 1

            self.scheduler.step()

            yield (
                None,
                None,
                rewards,
                losses,
                None,
                None,
                None,
            )
