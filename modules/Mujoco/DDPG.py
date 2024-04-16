import numpy as np
import torch.nn as nn
import os 
import sys
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

device = torch.device("cuda")

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
        self.action_memory = torch.zeros(self.capacity, 17, dtype=torch.float32).to(device)
        self.next_action_memory = torch.zeros(self.capacity, 17, dtype=torch.float32).to(device)
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
    

class Actor(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(Actor, self).__init__()

        self.inputs = n_inputs
        self.outputs = n_outputs
        self.hidden_dim = 3 * n_inputs

        self.fc1 = nn.Linear(self.inputs, self.hidden_dim)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc1)

        self.layernorm1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc2)

        self.layernorm2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(self.hidden_dim, self.outputs)
        nn.init.normal_(self.fc3.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc3)

        self.f = nn.GELU()

    def forward(self, state):
        x = self.f(self.layernorm1(self.fc1(state)))
        x = self.f(self.layernorm2(self.fc2(x)))

        return self.fc3(x).clamp(-0.4, 0.4)
    

class Critic(nn.Module):
    def __init__(self, n_inputs, n_actions):
        super(Critic, self).__init__()

        self.inputs = n_inputs
        self.actions = n_actions
        self.hidden_dim = 3 * n_inputs

        self.fc1 = nn.Linear(self.inputs, self.hidden_dim)
        nn.init.normal_(self.fc1.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc1)

        self.layernorm1 = nn.LayerNorm(self.hidden_dim)

        self.fc2 = nn.Linear(self.actions, self.hidden_dim)
        nn.init.normal_(self.fc2.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc2)

        self.layernorm2 = nn.LayerNorm(self.hidden_dim)

        self.fc3 = nn.Linear(2 * self.hidden_dim, 1)
        nn.init.normal_(self.fc3.weight, 0, 0.1)
        nn.utils.parametrizations.weight_norm(self.fc3)

        self.f = nn.GELU()

    def forward(self, state, actions):
        x1 = self.f(self.layernorm1(self.fc1(state)))
        x2 = self.f(self.layernorm1(self.fc2(actions)))

        x = torch.cat((x1, x2), dim=1)
        return self.fc3(x)
    

class Agent():
    def __init__(
        self, 
        env,
        lr,
        gamma,
        n_inputs,
        n_outputs,
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

        self.tau = 0.005

        self.actor1 = Actor(n_inputs, n_outputs).to(device)
        self.actor2 = Actor(n_inputs, n_outputs).to(device)
        self.actor2.load_state_dict(self.actor1.state_dict())

        self.optim1 = torch.optim.AdamW(self.actor1.parameters(), lr, weight_decay=1e-4)

        self.critic1 = Critic(n_inputs, n_outputs).to(device)
        self.critic2 = Critic(n_inputs, n_outputs).to(device)
        self.critic2.load_state_dict(self.critic1.state_dict())

        self.optim2 = torch.optim.AdamW(self.critic1.parameters(), lr*0.1, weight_decay=1e-4)
        self.memory = ReplayMemory(memory_size, 32, n_inputs)

    def update(self):

        self.actor1.train()
        self.critic1.eval()

        samples = self.memory.sample_batch()
        state = samples["states"]
        action = samples["actions"]
        next_state = samples["new_states"]
        reward = samples["rewards"].view(-1,1)
        next_action = samples["next_actions"].view(-1,1)
        done = samples["dones"].view(-1, 1)

        mask = 1 - done
        

        actions = self.actor1(state)
        Q_values = self.critic1(state, actions)
        loss = torch.mean(-Q_values)

        self.optim1.zero_grad()
        loss.backward()
        self.optim1.step()


        self.actor1.eval()
        self.actor2.eval()

        self.critic1.train()
        self.critic2.train()

        Q = self.critic1(state, action)
        A = self.actor2(next_state).detach()
        Q_n = self.critic2(next_state, A).detach()
        Q_t = reward + (Q_n * mask * self.gamma)

        critic_loss = torch.mean((Q_t - Q).pow(2))

        self.optim2.zero_grad()
        critic_loss.backward()
        self.optim2.zero_grad()

        for eval_param, target_param in zip(
            self.actor1.parameters(), self.actor2.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + eval_param.data * self.tau
            )
        
        for eval_param, target_param in zip(
            self.critic1.parameters(), self.critic2.parameters()
        ):
            target_param.data.copy_(
                target_param.data * (1.0 - self.tau) + eval_param.data * self.tau
            )

        return critic_loss

    def train(self, episodes, render):
        for e in range(episodes):
            done = False
            truncated = False

            state = self.env.reset()
            if (np.random.uniform(0, 1) >= self.epsilon):
                with torch.no_grad():
                    action = self.actor1(torch.FloatTensor(state).to(device))
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

                if isinstance(action, torch.Tensor):
                    next_state, reward, done, truncated = self.env.step(action.detach().cpu().resolve_conj().resolve_neg().numpy())
                else:
                    next_state, reward, done, truncated = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                if (np.random.uniform(0, 1) >= self.epsilon):
                    with torch.no_grad():
                        next_action = self.actor1(torch.FloatTensor(next_state).to(device))
                else:
                    next_action = self.env.action_space.sample()
                next_actions.append(next_action)
                self.memory.store_transition(state, action, reward, next_state, next_action, done)

                if len(self.memory) >= self.memory.capacity -1:
                    loss = self.update()
                    self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
                else:
                    loss = 1
                losses.append(loss)
                state = next_state
                action = next_action

                episode_reward += reward

            print(episode_reward)
            print(self.epsilon)
            print(e)
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
                    action = self.actor1(torch.FloatTensor(state).to(device))
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

                if isinstance(action, torch.Tensor):
                    next_state, reward, done, truncated = self.env.step(action.detach().cpu().resolve_conj().resolve_neg().numpy())
                else:
                    next_state, reward, done, truncated = self.env.step(action)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                next_states.append(next_state)
                dones.append(done)

                if (np.random.uniform(0, 1) >= self.epsilon):
                    with torch.no_grad():
                        next_action = self.actor1(torch.FloatTensor(next_state).to(device))
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
