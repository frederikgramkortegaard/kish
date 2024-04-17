import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from modules.Attentionsplit import AttentionSplit


class network(nn.Module):
    def __init__(self, heads, hidden_layers, inputs, outputs):
        super(network, self).__init__()

        self.encoder = AttentionSplit(
            inputs=inputs, hidden_size=hidden_layers, n_heads=4
        )
        self.decoder = AttentionSplit(
            inputs=hidden_layers, hidden_size=hidden_layers, n_heads=8
        )
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)
        self.f = nn.GELU()

    def forward(self, inputs):
        x, c = self.encoder(inputs)
        x, _ = self.decoder(x, c)
        return self.classi(x[:, -1, :])


class network2(nn.Module):
    def __init__(self, hidden_layers, inputs, outputs):
        super(network2, self).__init__()

        self.encoder = nn.LSTM(
            input_size=inputs, hidden_size=hidden_layers, batch_first=True, num_layers=1
        )
        self.decoder = nn.LSTM(
            input_size=hidden_layers,
            hidden_size=hidden_layers,
            batch_first=True,
            num_layers=1,
        )
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)

        self.f = nn.GELU()

    def forward(self, inputs):
        x, c = self.encoder(inputs)
        x, _ = self.decoder(x, c)
        return self.classi(x[:, -1, :])


class network3(nn.Module):
    def __init__(self, hidden_layers, inputs, outputs):
        super(network3, self).__init__()

        self.scale = nn.Linear(inputs, hidden_layers)
        self.encoder = nn.TransformerEncoderLayer(
            hidden_layers, 4, hidden_layers, activation="gelu", batch_first=True
        )
        self.decoder = nn.TransformerEncoderLayer(
            hidden_layers, 8, hidden_layers, activation="gelu", batch_first=True
        )
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)

        self.f = nn.GELU()

    def forward(self, inputs):
        inputs = self.f(self.scale(inputs))
        x = self.encoder(inputs)
        x = self.decoder(inputs)
        return self.classi(x[:, -1, :])


class agent:
    def __init__(self, data):
        self.network = network(4, 256, 4, 2).to(device)
        self.network2 = network2(256, 4, 2).to(device)
        self.network3 = network3(256, 4, 2).to(device)
        self.data = data
        self.optim = torch.optim.AdamW(
            self.network.parameters(),
            lr=0.0003,
            weight_decay=1e-4,
            amsgrad=True,
            fused=True,
        )
        self.optim2 = torch.optim.AdamW(
            self.network2.parameters(),
            lr=0.0003,
            weight_decay=1e-4,
            amsgrad=True,
            fused=True,
        )
        self.optim3 = torch.optim.AdamW(
            self.network3.parameters(),
            lr=0.0003,
            weight_decay=1e-4,
            amsgrad=True,
            fused=True,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def train(self):

        i = 0

        correct_list = []

        batched_states = []
        batched_next_actions = []

        for episode in self.data:

            states, _, _, _, _, _, next_actions = episode
            states = torch.stack(
                [torch.tensor(x, dtype=torch.float32, device=device) for x in states],
                dim=0,
            )

            next_actions = torch.stack(
                [torch.tensor(x, device=device) for x in next_actions], dim=0
            )

            split_states = states.split(5, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(5, 0)
            batched_next_actions.extend(split_actions)

        batched_states = torch.stack(batched_states, dim=0)
        batched_next_actions = torch.stack(batched_next_actions, dim=0)

        for l in range(5000):

            i += 1

            indexes = np.random.choice(len(batched_states) - 128, 128, replace=False)
            batch_state = batched_states[indexes]
            batch_action = batched_next_actions[indexes][:, -1]

            prediction = self.network(batch_state)

            loss = self.loss_fn(prediction, batch_action)

            self.optim.zero_grad()
            # nn.utils.clip_grad_value_(self.network.parameters(), 1)
            loss.backward()
            self.optim.step()

            prediction2 = self.network2(batch_state)

            loss2 = self.loss_fn(prediction2, batch_action)

            self.optim2.zero_grad()
            # nn.utils.clip_grad_value_(self.network.parameters(), 1)
            loss2.backward()
            self.optim2.step()

            prediction3 = self.network3(batch_state)

            loss3 = self.loss_fn(prediction3, batch_action)

            self.optim3.zero_grad()
            # nn.utils.clip_grad_value_(self.network.parameters(), 1)
            loss3.backward()
            self.optim3.step()

    def test(self, test_data):
        print("now testing")
        batched_states = []
        batched_next_actions = []

        self.network.eval()
        self.network2.eval()
        self.network3.eval()

        for episode in test_data:

            states, _, _, _, _, _, next_actions = episode
            states = torch.stack(
                [torch.tensor(x, dtype=torch.float32, device=device) for x in states],
                dim=0,
            )

            next_actions = torch.stack(
                [torch.tensor(x, device=device) for x in next_actions], dim=0
            )

            split_states = states.split(5, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(5, 0)
            batched_next_actions.extend(split_actions)

        batched_states = torch.stack(batched_states, dim=0)
        batched_next_actions = torch.stack(batched_next_actions, dim=0)

        attention = 0
        lstm = 0
        transformer = 0
        for l in range(25):
            total = 0
            correct = 0
            total2 = 0
            correct2 = 0
            total3 = 0
            correct3 = 0
            indexes = np.random.choice(len(batched_states) - 128, 128, replace=False)
            batch_state = batched_states[indexes]
            batch_action = batched_next_actions[indexes][:, -1]

            prediction = self.network(batch_state)

            _, predicted = torch.max(prediction.data, 1)
            total += batch_action.size(0)
            correct += (predicted == batch_action).sum().item()
            attention += correct / total
            print("Attention: ", correct / total)

            prediction2 = self.network2(batch_state)
            _, predicted2 = torch.max(prediction2.data, 1)
            total2 += batch_action.size(0)
            correct2 += (predicted2 == batch_action).sum().item()
            lstm += correct2 / total2
            print("lstm: ", correct2 / total2)

            prediction3 = self.network3(batch_state)
            _, predicted3 = torch.max(prediction3.data, 1)
            total3 += batch_action.size(0)
            correct3 += (predicted3 == batch_action).sum().item()
            transformer += correct3 / total3
            print("Transformer: ", correct3 / total3)

            print("")

        print("final Attention: ", attention / 25)
        print("final lstm: ", lstm / 25)
        print("final transformer: ", transformer / 25)
