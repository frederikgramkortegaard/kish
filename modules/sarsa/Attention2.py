import sys
import os 
import torch
import torch.nn as nn
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

device = torch.device('cuda')

from models.Atari.Attentionsplit import AttentionSplit


class network(nn.Module):
    def __init__(self, heads, hidden_layers, inputs, outputs):
        super(network, self).__init__()
        
        self.encoder = AttentionSplit(inputs=inputs, hidden_size=hidden_layers, n_heads=4)
        self.decoder = AttentionSplit(inputs=hidden_layers, hidden_size=hidden_layers, n_heads=64)
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)
        self.layernorm1 = nn.LayerNorm(hidden_layers)
        self.layernorm2 = nn.LayerNorm(hidden_layers)

        self.f = nn.GELU()
        

    def forward(self, inputs):
        x = self.f(self.layernorm1(self.encoder(inputs)))
        x = self.f(self.layernorm2(self.decoder(x)))
        return self.classi(x[:, -1, :])
    
class network2(nn.Module):
    def __init__(self, hidden_layers, inputs, outputs):
        super(network2, self).__init__()
        
        self.encoder = nn.LSTM(input_size=inputs, hidden_size=hidden_layers, batch_first=True, num_layers=1)
        self.decoder = nn.LSTM(input_size=hidden_layers, hidden_size=hidden_layers, batch_first=True, num_layers=1)
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)
        self.layernorm1 = nn.LayerNorm(hidden_layers)
        self.layernorm2 = nn.LayerNorm(hidden_layers)

        self.f = nn.GELU()
        

    def forward(self, inputs):
        x, c = self.encoder(inputs)
        x = self.f(self.layernorm1(x))
        x, c = self.decoder(x)
        x = self.f(self.layernorm2(x))
        return self.classi(x[:, -1, :])
    
class agent():
    def __init__(self, data):
        self.network = network(4, 256, 4, 2).to(device)
        self.network2 = network2(256, 4, 2).to(device)
        self.data = data
        self.optim = torch.optim.AdamW(self.network.parameters(), lr=0.0003, weight_decay=1e-4, amsgrad=True, fused=True)
        self.optim2 = torch.optim.AdamW(self.network2.parameters(), lr=0.0003, weight_decay=1e-4, amsgrad=True, fused=True)
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):

        i = 0

        correct_list = []

        batched_states = []
        batched_next_actions = []

        for episode in self.data:

            states, _, _, _, _, _, next_actions = episode
            states = torch.stack([torch.tensor(x, dtype=torch.float32, device=device) for x in states], dim=0)

            next_actions = torch.stack([torch.tensor(x, device=device) for x in next_actions], dim=0)

            split_states = states.split(5, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(5, 0)
            batched_next_actions.extend(split_actions)

        
        batched_states = torch.stack(batched_states, dim=0)
        batched_next_actions = torch.stack(batched_next_actions, dim=0)
        #print(batched_states.shape)
        #input()

        for l in range(5000):

            i += 1

            indexes = np.random.choice(len(batched_states)-128, 128, replace=False)
            batch_state = batched_states[indexes]
            batch_action = batched_next_actions[indexes][:, -1]

            prediction = self.network(batch_state)

            loss = self.loss_fn(prediction, batch_action)

            self.optim.zero_grad()
            #nn.utils.clip_grad_value_(self.network.parameters(), 1)
            loss.backward()
            self.optim.step()

            prediction2 = self.network2(batch_state)

            loss2 = self.loss_fn(prediction2, batch_action)

            self.optim2.zero_grad()
            #nn.utils.clip_grad_value_(self.network.parameters(), 1)
            loss2.backward()
            self.optim2.step()

    
    def test(self, test_data):
        print('now testing')
        batched_states = []
        batched_next_actions = []

        self.network.eval()
        self.network2.eval()

        for episode in test_data:

            states, _, _, _, _, _, next_actions = episode
            states = torch.stack([torch.tensor(x, dtype=torch.float32, device=device) for x in states], dim=0)

            next_actions = torch.stack([torch.tensor(x, device=device) for x in next_actions], dim=0)

            split_states = states.split(5, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(5, 0)
            batched_next_actions.extend(split_actions)

        batched_states = torch.stack(batched_states, dim=0)
        batched_next_actions = torch.stack(batched_next_actions, dim=0)

        for l in range(10):
            total = 0
            correct = 0
            total2 = 0
            correct2 = 0
            indexes = np.random.choice(len(batched_states)-32, 32, replace=False)
            batch_state = batched_states[indexes]
            batch_action = batched_next_actions[indexes][:, -1]

            prediction = self.network(batch_state)

            _, predicted = torch.max(prediction.data, 1)
            total += batch_action.size(0)
            correct += (predicted == batch_action).sum().item()
            print('Attention: ', correct / total)

            prediction2 = self.network2(batch_state)
            _, predicted2 = torch.max(prediction2.data, 1)
            total2 += batch_action.size(0)
            correct2 += (predicted2 == batch_action).sum().item()
            print('lstm: ', correct2 / total2)
