import torch
import torch.nn as nn
import numpy as np
import gym
from skimage.color import rgb2gray
from matplotlib import pyplot as plt
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

        self.state_memory = torch.zeros((self.capacity, frames, height, width), dtype=torch.float32).to(device)
        self.new_state_memory = torch.zeros((self.capacity, frames, height, width), dtype=torch.float32)
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
        self.size +=1
    
    # returns a sample using indexes
    def sample_batch(self):

        indices = np.random.choice(self.capacity-self.batch_size, size=self.batch_size, replace=False)
        return dict ( states=self.state_memory[indices], new_states=self.new_state_memory[indices], actions=self.action_memory[indices],
                      rewards = self.reward_memory[indices], dones = self.done_memory[indices], indices = indices )

    def __len__(self):
        return self.size

class f(nn.Module):
    def __init__(self):
        super(f, self).__init__()

    def forward(self, x):
        return torch.sinh(nn.functional.sigmoid(x)) * x # Generalized Gaussian Error unit

class Network(nn.Module):
    def __init__(self, inputs, actions):
        super(Network, self).__init__()
        self.input = inputs
        self.actions = actions

        self.features = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            f(),
            nn.MaxUnpool2d(3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3, groups=2),
            nn.BatchNorm2d(32),
            f(),
            nn.MaxUnpool2d(3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=2, groups=2),
            nn.BatchNorm2d(64),
            f(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding=2, groups=2),
            nn.BatchNorm2d(64),
            f(),
            nn.AvgPool2d(3),
        )

        for module in self.features:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2. / n))

        self.dropout = nn.Dropout(0.5)
        self.action = nn.Sequential(
            nn.Linear(256, 256),
            nn.LayerNorm(256),
            f(),
            nn.AlphaDropout(0.5),
            nn.Linear(256, self.actions),
        )
        for module in self.action:
            if isinstance(module, nn.LayerNorm):
                nn.init.normal_(module.weight)
            if isinstance(module, nn.Linear):
                nn.init.orthogonal_(module.weight)

        self.f = f()

        self.to(device)

    def forward(self, states):
        states = states.view(states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3])

        xs = []
        i = 0
        while i != states.shape[1]:
            state = states[:, i, :, :]
            x = self.dropout(self.features(state))
            xs.append(x.flatten(1))
            i += 1
        x = torch.stack(xs, dim=1)
        print(x.shape)
        input()
        x = self.dropout(self.encoder(x))
        return self.action(x[:, -1, :])

class Agent():

    def __init__(self):
        self.game = gym.make('SpaceInvaders-v0')
        state = cv2.resize(rgb2gray(crop(self.game.reset(), ((13, 13), (15, 25), (0, 0)))), (84, 84))
        self.steps = 0
        self.frames = 3
        self.game.seed(0)
        self.eval = Network(state.flatten().shape[0], self.game.action_space.n).to(device)
        self.eval_target = Network(state.flatten().shape[0], self.game.action_space.n).to(device)
        self.eval_target.load_state_dict(self.eval.state_dict())

        model_parameters = filter(lambda p: p.requires_grad, self.eval.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        print(params)
        input()
        self.memory = ReplayMemory(50000, 16, self.frames, state.shape[0], state.shape[1], 12, 0.99)
        self.optimizer = OrthAdam(self.eval.parameters(), lr=0.00003, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(self.optimizer, 100, 2)
        self.transform = transforms.Compose([
            transforms.GaussianBlur([11, 11], (0.1, 3)),
            transforms.RandomErasing(),
            transforms.RandomRotation(degrees=[-180, 180]),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

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

        loss = ((Q_value - Q_target)**2).mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        for eval_param, target_param in zip(self.eval.parameters(), self.eval_target.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - self.TAU) + eval_param.data * self.TAU)

        self.steps += 1

        return loss

    def stack_frames(self, stacked_frames, state, is_new_episode, max_frames):

        if is_new_episode:
            stacked_frames = deque(
                [np.zeros((1, state.shape[0], state.shape[1]), dtype=np.float64) for _ in range(self.frames)],
                maxlen=max_frames,
            )

            stacked_frames.extend([state] * max_frames)
            stacked_state = np.stack(stacked_frames, axis=0)

        else:

            stacked_frames.append(state)
            stacked_state = np.stack(stacked_frames, axis=0)

        return torch.FloatTensor(stacked_state), stacked_frames
        
    def train(self, episodes, render):

        episode_avg_rew = []
        episode_avg_loss = []
        episode_counter = []
        episode_avg_mean_action = []
        episode_avg_std_action = []
        episode_avg_mean = []
        episode_avg_std = []

        for i in range(episodes):
            if i == 0:
                i = 1
            if self.epsilon == 1:
                episode_avg_rew = []
                episode_avg_loss = []
                episode_counter = []
                episode_avg_mean_action = []
                episode_avg_std_action = []
                episode_avg_mean = []
                episode_avg_std = []

            state = cv2.resize(rgb2gray(crop(self.game.reset(), ((13, 13), (15, 25), (0, 0)))), (84, 84))

            stacked_frames = deque(
            [np.zeros((1, state.shape[0], state.shape[1]), dtype=np.float64) for _ in range(self.frames)],
            maxlen=self.frames,
            )
            state, stacked_frames = self.stack_frames(
                stacked_frames, state, True, self.frames
            )

            done = False
            truncated = False

            episode_reward = 0
            episode_loss = 0
            episode_mean_action = 0
            episode_std_action = 0
            episode_mean = 0
            episode_std = 0
            self.lives = self.game.ale.lives()
            j = 0

            while (not done or not truncated):
                
                if len(self.memory) > (50000) and render == True:
                    self.game.render()

                if j % 3 == 0:
                    if (np.random.uniform(0, 1) >= self.epsilon) and len(self.memory) > (50000):
                        with torch.no_grad():
                            self.eval.eval()
                            action_choice = self.eval(torch.FloatTensor(state).unsqueeze_(0).to(device))
                            episode_mean_action += torch.mean(action_choice).item()
                            episode_std_action += torch.std(action_choice).item()
                            action = torch.argmax(action_choice).item()
                    else:
                        action_choice = np.random.normal(size=self.game.action_space.n)
                        action = np.argmax(action_choice)
                        

                next_state, reward, done, truncated = self.game.step(action)
                true_reward = reward

                reward = min(1, reward)

                if self.lives > self.game.ale.lives():
                    reward -= 1
                    self.lives = self.game.ale.lives()

                next_state, stacked_frames = self.stack_frames(
                stacked_frames, cv2.resize(rgb2gray(crop(next_state, ((13, 13), (15, 25), (0, 0)))), (84, 84)), False, self.frames
                )

                self.memory.store_transition(state, next_state, action, reward, done)

                if len(self.memory) > (50) and j % 3 == 0:
                    current_loss = self.update()
                    episode_loss += current_loss.item()
                    self.epsilon = max(self.epsilon * 0.999995, 0.01)

                state = next_state

                episode_mean += torch.mean(self.eval.action[-1].weight).item()
                episode_std += torch.std(self.eval.action[-1].weight).item()
                episode_reward += true_reward
                j += 1

            self.scheduler.step()
            episode_avg_rew.append(episode_reward)
            episode_avg_loss.append((episode_loss*3) / j)
            episode_avg_mean.append(episode_mean / j)
            episode_avg_std.append(episode_std / j)
            episode_avg_mean_action.append((episode_mean_action*3) / j)
            episode_avg_std_action.append((episode_std_action*3) / j)
            episode_counter.append(i)
            print('episode')
            print(i)
            print('avg_reward')
            print(sum(episode_avg_rew) / len(episode_avg_rew))
            print('avg_loss')
            print(sum(episode_avg_loss) / len(episode_avg_loss))
            print('avg_mean')
            print(sum(episode_avg_mean) / len(episode_avg_mean))
            print('avg_std')
            print(sum(episode_avg_std) / len(episode_avg_std))
            print('avg_mean_action')
            print(sum(episode_avg_mean_action) / len(episode_avg_mean_action))
            print('avg_std_action')
            print(sum(episode_avg_std_action) / len(episode_avg_std_action))
            print('epsilon')
            print(self.epsilon)
            if i == 59999:
                plt.plot(episode_counter, episode_avg_rew)
                plt.show()

            yield (
                state,
                action,
                reward,
                current_loss,
                next_state,
                done,
                None,
            )