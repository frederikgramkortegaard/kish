import torch
import torch.nn as nn
import numpy as np
from skimage.color import rgb2gray
from collections import deque
import cv2
import math
from skimage.util import crop
import sys
import torch.distributions as dist
import os

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.Optimizer import OrthAdam
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
        n_outputs,
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
        self.action_memory = torch.zeros(self.capacity, n_outputs, dtype=torch.float32)
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
        reward = torch.tensor(reward).to(device).clamp(0, 1)
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


class Actor(nn.Module):
    def __init__(self, actions):
        super(Actor, self).__init__()
        self.actions = actions

        self.logstd = nn.Parameter(torch.zeros(1, actions))
        self.logstd.data.normal_(-9, 0.001)
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
            nn.MaxPool2d(3),
        )

        for module in self.vision:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n))  # Microsoft Init

        # Forward sequence
        self.encoder = AttentionSplit(576, 576, 8)

        self.action = VBLinear(576, self.actions)

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

        return self.action(x[:, -1, :]), self.logstd.clamp(-11, 11).exp(), c


class Critic(nn.Module):
    def __init__(self, n_actions, hidden_dim, frames):
        super(Critic, self).__init__()

        self.actions = n_actions
        self.hidden_dim = hidden_dim

        self.vision = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=16, kernel_size=11, padding=5),
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
            nn.MaxPool2d(3),
        )

        for module in self.vision:
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                nn.init.normal_(module.weight, 0, math.sqrt(2.0 / n))  # Microsoft Init

        self.fc1 = nn.Linear(576 + n_actions, self.hidden_dim)
        n1 = self.fc1.weight.shape[1] * self.fc1.weight.shape[0]
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2 / n1))

        self.layernorm1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, 1)
        n2 = self.fc2.weight.shape[1] * self.fc2.weight.shape[0]
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2 / n2))

        self.f = nn.GELU()

    def forward(self, state, actions):
        x = self.vision(state[:, -3:, :, :]).flatten(1)
        x = torch.cat((x, actions), dim=1)
        x = self.f(self.layernorm1(self.fc1(x)))
        return self.fc2(x)


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
    ):

        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.frames = frames
        self.hidden_dim = hidden_dim
        self.tau = 0.005
        self.temp = nn.Parameter(torch.tensor(0.2))
        self.temp.to(device)
        self.entropy_target = -torch.tensor(n_outputs)

        self.actor = Actor(n_outputs).to(device)
        self.optim1 = torch.optim.AdamW(
            self.actor.parameters(),
            lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )

        self.critic1 = Critic(n_outputs, self.hidden_dim, frames).to(device)
        self.critic2 = Critic(n_outputs, self.hidden_dim, frames).to(device)

        self.critic1_target = Critic(n_outputs, self.hidden_dim, frames).to(device)
        self.critic2_target = Critic(n_outputs, self.hidden_dim, frames).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.optim2 = torch.optim.AdamW(
            self.critic1.parameters(),
            lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )
        self.optim3 = torch.optim.AdamW(
            self.critic2.parameters(),
            lr,
            weight_decay=weight_decay,
            amsgrad=True,
        )
        self.optim4 = torch.optim.AdamW(
            [self.temp], lr, weight_decay=weight_decay, amsgrad=True
        )
        self.memory = ReplayMemory(
            capacity=mem_size,
            batch_size=batch_size,
            frames=frames,
            n_step=n_step,
            hidden_dim=self.hidden_dim,
            gamma=self.gamma,
            height=height,
            width=width,
            n_outputs=n_outputs,
        )

    def update_critic_(
        self,
        critic,
        optim,
        state,
        action,
        reward,
        next_state,
        mask,
        a_log_prob,
        actions,
    ):

        Q_values = critic(state, action)

        with torch.no_grad():

            q_1 = self.critic1_target(next_state, actions)
            q_2 = self.critic2_target(next_state, actions)

            q = torch.min(q_1, q_2)

            v = mask * (q - self.temp * a_log_prob)
            Q_target = reward + self.gamma * v

        loss = torch.mean((Q_values - Q_target).pow(2))  # Equation 5 from SAC

        optim.zero_grad()
        loss.backward()
        optim.step()

    def update(self):

        samples = self.memory.sample_batch()
        state = samples["states"].to(device)
        action = samples["actions"].to(device)
        next_state = samples["new_states"].to(device)
        reward = samples["rewards"].view(-1, 1).to(device)
        done = samples["dones"].view(-1, 1).to(device)
        hiddens = samples["hiddens"].to(device)

        mask = 1 - done

        with torch.no_grad():
            _, _, hiddens_ = self.actor(state, hiddens)

        a_, std_, _ = self.actor(next_state, hiddens_)
        probs_ = dist.Normal(a_, std_)
        u_ = probs_.rsample()
        a_log_prob_ = probs_.log_prob(u_)
        actions_ = torch.tanh(u_)
        a_log_prob_ -= torch.log(1 - actions_**2 + 1e-8)

        self.update_critic_(
            self.critic1,
            self.optim2,
            state,
            action,
            reward,
            next_state,
            mask,
            a_log_prob_.detach(),
            actions_.detach(),
        )
        self.update_critic_(
            self.critic2,
            self.optim3,
            state,
            action,
            reward,
            next_state,
            mask,
            a_log_prob_.detach(),
            actions_.detach(),
        )

        q_1 = self.critic1(next_state, actions_)
        q_2 = self.critic2(next_state, actions_)
        q = torch.min(q_1, q_2)

        loss = (
            self.temp.detach() * a_log_prob_ - q
        ).mean()  # Update with equation 9 from SAC paper

        self.optim1.zero_grad()
        loss.backward()
        self.optim1.step()

        alpha_loss = (
            -self.temp * (a_log_prob_ + self.entropy_target).detach()
        ).mean()  # Update alpha / tempature parameter with equation 18 from SAC paper

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

        return torch.FloatTensor(stacked_state), stacked_frames

    def train(self, timesteps, render):
        t = 0
        rewards = []
        losses = []
        while t <= timesteps:

            done = False
            truncated = False
            state, _ = self.env.reset()
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
            hiddens = torch.zeros((self.frames, self.hidden_dim)).to(device)
            prev_hiddens = hiddens.clone()
            while not done and not truncated:

                if j % 3 == 0:
                    with torch.no_grad():
                        self.actor.eval()
                        a, std, hiddens = self.actor(
                            torch.FloatTensor(state).to(device).unsqueeze(0), hiddens
                        )
                        probs = dist.Normal(a, std)
                        u = probs.rsample()
                        action = torch.tanh(u)
                        self.actor.train()

                if render and len(self.memory) > 1000:
                    self.env.render()

                next_state, reward, done, truncated, _ = self.env.step(
                    action.argmax(dim=-1).item()
                )

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
                self.memory.store_transition(
                    state=state,
                    action=action,
                    reward=reward,
                    new_state=next_state,
                    done=done,
                    hidden=prev_hiddens,
                )
                prev_hiddens = hiddens

                if len(self.memory) > 1000 and j % 3 == 0:
                    loss = self.update().item()
                else:
                    loss = 1
                losses.append(loss)
                state = next_state

                if (t % 100) == 0:
                    print("timestep: ", t)
                    print("Reward: ", sum(rewards))
                    print("Loss: ", sum(losses))
                    print("Average reward: ", sum(rewards) / len(rewards))
                    print("Average loss: ", sum(losses) / len(losses))
                    print()
                if (t % 1000) == 0:
                    yield (rewards, losses)
                    rewards = []
                    losses = []

                j += 1
                t += 1
