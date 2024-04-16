import numpy as np
import torch.nn as nn
import os 
import sys
import cv2
from skimage.util import crop
from skimage.color import rgb2gray
import torch
import math
import torch.distributions as dist
from collections import deque
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.Atari.Attentionsplit import AttentionSplit
from models.Atari.Optimizer import AdamP2
device = torch.device("cuda")

# Class for storing transition memories
# also includes functions for batching, both randomly and according to index, which we need for nstep learning
class ReplayMemory:
    def __init__(self, capacity, batch_size, frames, height, width, n_step, gamma):
        self.capacity = capacity
        self.mem_counter = 0
        self.gamma = gamma
        self.mem_size = 0

        self.state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        ).to(device)
        self.new_state_memory = torch.zeros(
            (self.capacity, frames, height, width), dtype=torch.float32
        )
        self.action_memory = torch.zeros(self.capacity, 6, dtype=torch.float32).to(device)
        self.reward_memory = torch.zeros(self.capacity,  dtype=torch.float32).to(device)
        self.terminal_memory = torch.zeros(self.capacity, dtype=torch.float32).to(device)
        self.max_size = self.capacity
        self.batch_size = batch_size
        self.index = 0
        self.size = 0 
        self.n_mem = deque(maxlen=n_step)


    # Stores a transition, while checking for the n-step value passed 
    # if the n-step buffer is not larger than n, we only store the transition there
    def store_transition(self, state, action, reward, state_, done):

        self.n_mem.append(reward)
        if len(self.n_mem) < self.n_mem.maxlen:
            return

        # Calculate n_step reward
        temp = self.n_mem.copy()
        temp.reverse()
        for rew in temp:
            reward = reward + self.gamma * rew

        
        state = torch.tensor(state).to(device)
        action = torch.tensor(action).to(device)
        reward = torch.tensor(reward).to(device)
        state_ = torch.tensor(state_).to(device)
        
        if done == False:
            done = 0
        else:
            done = 1
        done = torch.tensor(done).to(device)

        self.state_memory[self.index] = state
        self.action_memory[self.index] = action
        self.new_state_memory[self.index] = state_
        self.reward_memory[self.index] = reward
        self.terminal_memory[self.index] = done
        
        self.index = (self.index + 1) % self.capacity
        self.size +=1

    # returns a random sample using indexes
    def sample_batch(self):

        indices = np.random.choice(self.capacity, size=self.batch_size, replace=False)

        return dict ( states=self.state_memory[indices], actions=self.action_memory[indices], new_states=self.new_state_memory[indices],
                      rewards = self.reward_memory[indices], dones=self.terminal_memory[indices], indices = indices, )

    def __len__(self):
        return self.size
    

class Actor(nn.Module):
    def __init__(self, actions):
        super(Actor, self).__init__()
        self.actions = actions

        self.logstd = nn.Parameter(torch.zeros(1, actions))

        self.vision = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3, groups=8),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, groups=16),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.MaxPool2d(3),
        )

        for module in self.vision:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init

        self.dropout = nn.Dropout(0.5)
        self.input_dropout = nn.Dropout(0.2)

        # Forward sequence 
        self.forward_one = AttentionSplit(576, 576, 8)
        self.forward_two = AttentionSplit(576, 576, 8)

        self.action = nn.Linear(576, self.actions)

        self.to(device)

    def forward(self, states):
        states = states.view(states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3])
        # Assuming states has shape (batch_size, num_frames, num_channels, height, width)
        batch_size, num_frames, _, _, _ = states.shape

        # Reshape states to combine batch_size and num_frames dimensions
        states = self.input_dropout(states.view(-1, states.shape[2], states.shape[3], states.shape[4]))

        x = self.dropout(self.vision(states))

        # Reshape back to (batch_size, num_frames, -1)
        x = x.view(batch_size, num_frames, -1)
        
        x, c = self.forward_one(x)
        x, _ = self.forward_two(x, c)
        x = self.dropout(x[:, -1, :])

        return self.action(x), torch.exp(self.logstd)

class Critic(nn.Module):
    def __init__(self, actions):
        super(Critic, self).__init__()
        self.actions = actions

        self.vision = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=11, padding=5),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=7, padding=3, groups=8),
            nn.GroupNorm(8, 32),
            nn.GELU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1, groups=16),
            nn.GroupNorm(16, 64),
            nn.GELU(),
            nn.MaxPool2d(3),
        )

        for module in self.vision:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n)) # Microsoft Init

        self.dropout = nn.Dropout(0.5)
        self.input_dropout = nn.Dropout(0.2)

        # Forward sequence 
        self.forward_one = AttentionSplit(576, 576, 8)
        self.forward_two = AttentionSplit(576, 576, 8)

        self.action = nn.Linear(self.actions, 256)

        self.final = nn.Linear(576 + 256, 1)

        self.to(device)

    def forward(self, states, actions):
        states = states.view(states.shape[0], states.shape[1], 1, states.shape[2], states.shape[3])
        # Assuming states has shape (batch_size, num_frames, num_channels, height, width)
        batch_size, num_frames, _, _, _ = states.shape

        # Reshape states to combine batch_size and num_frames dimensions
        states = self.input_dropout(states.view(-1, states.shape[2], states.shape[3], states.shape[4]))

        x = self.dropout(self.vision(states))

        # Reshape back to (batch_size, num_frames, -1)
        x = x.view(batch_size, num_frames, -1)
        
        x, c = self.forward_one(x)
        x, _ = self.forward_two(x, c)
        x = self.dropout(x[:, -1, :])

        x2 = self.action(actions)
        x = torch.cat((x, x2), dim=1)
        return self.final(x)
    

class Agent():
    def __init__(
        self, 
        env,
        lr,
        gamma,
        epsilon,
        epsilon_decay,
        epsilon_min,
        batch_size,
        memory_size,
        frames,
        n_step,
        weight_decay,
    ):
        self.env = env
        state = cv2.resize(
            rgb2gray(crop(self.env.reset(), ((13, 13), (15, 25), (0, 0)))), (84, 84)
        )
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.frames = frames
        self.f = nn.GELU()

        self.tau = 0.005
        self.temp = nn.parameter.Parameter(torch.tensor(0.2))
        self.temp.to(device)
        self.entropy_target = - torch.tensor(env.action_space.n)

        self.actor = Actor(env.action_space.n).to(device)
        self.optim1 = AdamP2(self.actor.parameters(), lr, weight_decay=weight_decay)

        self.critic1 = Critic(env.action_space.n).to(device)
        self.critic2 = Critic(env.action_space.n).to(device)
        self.critic1_target = Critic(env.action_space.n).to(device)
        self.critic2_target = Critic(env.action_space.n).to(device)
        self.critic2_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.optim2 = AdamP2(self.critic1.parameters(), lr, weight_decay=weight_decay)
        self.optim3 = AdamP2(self.critic2.parameters(), lr, weight_decay=weight_decay)
        self.optim4 = AdamP2([self.temp], lr, weight_decay=weight_decay)
        self.memory = ReplayMemory(memory_size, batch_size, frames, state.shape[0], state.shape[1], n_step, self.gamma)

    def update_critic_(self, critic, optim, state, action, reward, next_state, mask):
        Q_values = critic(state, action)

        with torch.no_grad():
            a, std = self.actor(next_state)
            probs = dist.Normal(a, std)
            u = probs.rsample()
            a_log_prob = probs.log_prob(u)
            next_action = torch.tanh(u)
            a_log_prob -= torch.sum(torch.log(1 - next_action**2))

            q_1 = self.critic1_target(next_state, next_action)
            q_2 = self.critic2_target(next_state, next_action)

            q = torch.min(q_1, q_2)

            v = mask * (q - self.temp * a_log_prob)
            Q_target = reward + self.gamma * v

        loss = torch.mean(0.5 * (Q_values - Q_target).pow(2)) # Equation 5 from SAC

        optim.zero_grad()
        loss.backward()
        optim.step()

    def update(self):

        samples = self.memory.sample_batch()
        state = samples["states"]
        action = samples["actions"]
        next_state = samples["new_states"].to(device)
        reward = samples["rewards"].view(-1,1)
        done = samples["dones"].view(-1, 1)

        mask = 1 - done

        self.update_critic_(self.critic1, self.optim2, state, action, reward, next_state, mask)
        self.update_critic_(self.critic2, self.optim3, state, action, reward, next_state, mask)
        
        a, std = self.actor(state)
        probs = dist.Normal(a, std)
        u = probs.rsample()
        a_log_prob = probs.log_prob(u)
        actions = torch.tanh(u)
        a_log_prob -= torch.sum(torch.log(1 - actions**2))

        q_1 = self.critic1(next_state, actions)
        q_2 = self.critic2(next_state, actions)
        q = torch.min(q_1, q_2)

        loss = (self.temp.detach() * a_log_prob - q).mean() # Update with equation 9 from SAC paper

        self.optim1.zero_grad()
        loss.backward()
        self.optim1.step()

        alpha_loss = (-self.temp * a_log_prob.detach() - self.temp * self.entropy_target).mean() # Update alpha / tempature parameter with equation 18 from SAC paper

        self.optim4.zero_grad()
        alpha_loss.backward()
        self.optim4.step()

        for eval_param, target_param in zip(
            self.critic1.parameters(), self.critic1_target.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + eval_param.data * self.tau
            )
        
        for eval_param, target_param in zip(
            self.critic2.parameters(), self.critic2_target.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + eval_param.data * self.tau
            )

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
        reward_sum = []
        for e in range(episodes):
            done = False
            truncated = False

            state = cv2.resize(
            rgb2gray(crop(self.env.reset(), ((13, 13), (15, 25), (0, 0)))), (84, 84)
            )
            episode_reward = 0

            states = []
            actions = []
            rewards = []
            losses = []
            next_states = []
            dones = []
            next_actions = []

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

            self.lives = self.env.ale.lives()
            j = 0
            while not done or not truncated:
                
                if j % 3 == 0:
                    if (np.random.uniform(0, 1) >= self.epsilon) and len(self.memory) > (self.memory.capacity):
                        with torch.no_grad():
                            a, std = self.actor(torch.FloatTensor(state.unsqueeze(0)).to(device))
                            probs = dist.Normal(a, std)
                            action = probs.rsample()
                    else:
                        action = torch.normal(0, 1, size=(1, self.env.action_space.n))


                if render and len(self.memory) >= self.memory.capacity -1:
                    self.env.render()

                next_state, reward, done, truncated = self.env.step(action.argmax().detach().cpu().resolve_conj().resolve_neg().numpy())

                next_state, stacked_frames = self.stack_frames(
                    stacked_frames,
                    cv2.resize(
                        rgb2gray(crop(next_state, ((13, 13), (15, 25), (0, 0)))),
                        (84, 84),
                    ),
                    False,
                    self.frames,
                )

                states.append(state)
                actions.append(action)
                next_states.append(next_state)
                dones.append(done)
                true_reward = reward

                reward = min(1, reward) 

                rewards.append(true_reward)

                self.memory.store_transition(state, action, reward, next_state, done)

                if len(self.memory) > (self.memory.capacity) and j % 3 == 0:
                    loss = self.update().item()
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                else:
                    loss = 1
                losses.append(loss)
                state = next_state
                episode_reward += true_reward
                j += 1

            reward_sum.append(episode_reward)
            print(sum(reward_sum) / len(reward_sum))
            print(self.epsilon)
            print(sum(losses) / len(losses))
            print(e)
            print()
            yield (
                states,
                actions,
                rewards,
                losses,
                next_states,
                dones,
                next_actions,
            )

