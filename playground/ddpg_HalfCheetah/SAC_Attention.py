import numpy as np
import torch.nn as nn
import os
import sys
import torch
import torch.distributions as dist
import math
from collections import deque

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from modules.Attentionsplit import AttentionSplit
from modules.Optimizer import RNAdamWP

device = torch.device("cuda")


# Class for storing transition memories
# also includes functions for batching, both randomly and according to index, which we need for nstep learning
class ReplayMemory:
    def __init__(
        self, capacity, batch_size, input, n_outputs, frames, n_hidden, n_step, gamma
    ):
        self.capacity = capacity
        self.mem_counter = 0
        self.gamma = gamma
        self.mem_size = 0
        self.n_hidden = n_hidden
        self.frames = frames
        self.n_step = n_step

        self.state_memory = torch.zeros(
            (self.capacity, frames, input), dtype=torch.float32
        ).to(device)
        self.new_state_memory = torch.zeros(
            (self.capacity, frames, input), dtype=torch.float32
        ).to(device)
        self.action_memory = torch.zeros(
            self.capacity, n_outputs, dtype=torch.float32
        ).to(device)
        self.hidden_memmory = torch.zeros(
            self.capacity, frames, n_hidden, dtype=torch.float32
        ).to(device)
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
    def store_transition(self, state, action, reward, state_, done, hidden):

        hidden = hidden.view(self.frames, self.n_hidden)

        reward = torch.tensor(reward).to(device)

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
        self.hidden_memmory[self.index] = hidden

        if self.size < self.capacity - 1:
            self.size += 1
            self.index += 1
        else:
            self.index = (self.index + 1) % self.capacity

    # returns a random sample using indexes
    def sample_batch(self):

        indices = np.random.choice(self.size, size=self.batch_size, replace=False)

        return dict(
            states=self.state_memory[indices],
            actions=self.action_memory[indices],
            new_states=self.new_state_memory[indices],
            rewards=self.reward_memory[indices],
            dones=self.terminal_memory[indices],
            indices=indices,
            hiddens=self.hidden_memmory[indices],
        )

    def __len__(self):
        return self.size


class Actor(nn.Module):
    def __init__(self, n_inputs, n_outputs, hidden_dim, frames):
        super(Actor, self).__init__()

        self.inputs = n_inputs
        self.outputs = n_outputs
        self.hidden_dim = hidden_dim

        self.logstd = nn.Parameter(torch.zeros(1, n_outputs))
        self.logstd.data.normal_(math.sqrt(2 / self.logstd.numel()), 0.01)

        self.fc1 = nn.Linear(self.inputs, 32)
        n1 = self.fc1.weight.shape[1] * self.fc1.weight.shape[0]
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2 / n1))

        self.layernorm1 = nn.LayerNorm(32)

        self.encoder1 = AttentionSplit(32, self.hidden_dim, 16)
        self.encoder2 = AttentionSplit(self.hidden_dim, self.hidden_dim, 16)

        self.fc2 = nn.Linear(self.hidden_dim, self.outputs)
        n2 = self.fc2.weight.shape[1] * self.fc2.weight.shape[0]
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2 / n2))

        self.f = nn.GELU()

    def forward(self, state, hidden=None):
        bs, ls, fs = state.shape
        state = state.view(-1, fs)
        x = self.f(self.layernorm1(self.fc1(state)))
        x = x.view(bs, ls, 32)

        x, c = self.encoder1(x, hidden)
        x, c = self.encoder2(x, c)

        return self.fc2(x[:, -1, :]), torch.exp(self.logstd), c


class Critic(nn.Module):
    def __init__(self, n_inputs, n_actions, hidden_dim):
        super(Critic, self).__init__()

        self.inputs = n_inputs
        self.actions = n_actions
        self.hidden_dim = hidden_dim

        self.fc1 = nn.Linear(self.inputs + n_actions, self.hidden_dim)
        n1 = self.fc1.weight.shape[1] * self.fc1.weight.shape[0]
        nn.init.normal_(self.fc1.weight, 0, math.sqrt(2 / n1))

        self.layernorm1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, 1)
        n2 = self.fc2.weight.shape[1] * self.fc2.weight.shape[0]
        nn.init.normal_(self.fc2.weight, 0, math.sqrt(2 / n2))

        self.f = nn.GELU()

    def forward(self, state, actions):
        x = torch.cat((state[:, -1, :], actions.flatten(1)), dim=1)
        x = self.f(self.layernorm1(self.fc1(x)))
        return self.fc2(x)


class Agent:
    def __init__(
        self,
        env,
        lr,
        gamma,
        n_inputs,
        n_outputs,
        batch_size,
        memory_size,
        frames,
        hidden_dim,
        n_step,
    ):
        self.env = env
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.memory_size = memory_size
        self.frames = frames

        self.hidden_dim = max(n_inputs, hidden_dim)
        self.tau = 0.005
        self.temp = nn.Parameter(torch.tensor(0.2, device=device))
        self.temp.to(device)
        self.entropy_target = -torch.tensor(n_outputs)

        self.actor = Actor(n_inputs, n_outputs, self.hidden_dim, self.frames).to(device)
        self.optim1 = RNAdamWP(self.actor.parameters(), lr)

        self.critic1 = Critic(n_inputs, n_outputs, self.hidden_dim).to(device)
        self.critic2 = Critic(n_inputs, n_outputs, self.hidden_dim).to(device)

        self.critic1_target = Critic(n_inputs, n_outputs, self.hidden_dim).to(device)
        self.critic2_target = Critic(n_inputs, n_outputs, self.hidden_dim).to(device)

        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())

        self.optim2 = RNAdamWP(self.critic1.parameters(), lr)
        self.optim3 = RNAdamWP(self.critic2.parameters(), lr)
        self.optim4 = RNAdamWP([self.temp], lr)

        self.memory = ReplayMemory(
            capacity=memory_size,
            batch_size=batch_size,
            input=n_inputs,
            n_outputs=n_outputs,
            frames=frames,
            n_hidden=self.hidden_dim,
            n_step=n_step,
            gamma=self.gamma,
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
        state = samples["states"]
        action = samples["actions"]
        next_state = samples["new_states"].to(device)
        reward = samples["rewards"].view(-1, 1)
        done = samples["dones"].view(-1, 1)
        hiddens = samples["hiddens"]

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

                with torch.no_grad():
                    self.actor.eval()
                    a, std, hiddens = self.actor(
                        torch.FloatTensor(state).to(device).unsqueeze(0), hiddens
                    )
                    probs = dist.Normal(a, std)
                    u = probs.rsample()
                    action = torch.tanh(u)
                    self.actor.train()

                if render and len(self.memory) > 10000:
                    self.env.render()

                next_state, reward, done, truncated, _ = self.env.step(
                    action.flatten(0)
                    .detach()
                    .cpu()
                    .resolve_conj()
                    .resolve_neg()
                    .numpy()
                )

                next_state, stacked_frames = self.stack_frames(
                    stacked_frames,
                    next_state,
                    False,
                    self.frames,
                )

                rewards.append(reward)
                self.memory.store_transition(
                    state, action, reward, next_state, done, prev_hiddens
                )
                prev_hiddens = hiddens

                if len(self.memory) > 10000:
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
