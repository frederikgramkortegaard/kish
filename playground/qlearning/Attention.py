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

        self.scale = nn.Linear(inputs, hidden_layers)
        self.encoder = AttentionSplit(
            inputs=hidden_layers, hidden_size=hidden_layers, n_heads=4
        )
        self.decoder = AttentionSplit(
            inputs=hidden_layers, hidden_size=hidden_layers, n_heads=8
        )
        self.classi = nn.Linear(hidden_layers, outputs)

        nn.init.normal_(self.classi.weight, std=0.1)
        self.f = nn.GELU()

    def forward(self, inputs):
        inputs = self.f(self.scale(inputs))
        x, c = self.encoder(inputs)
        x, _ = self.decoder(x, c)
        return self.classi(x[:, -1, :])


class network2(nn.Module):
    def __init__(self, hidden_layers, inputs, outputs):
        super(network2, self).__init__()

        self.scale = nn.Linear(inputs, hidden_layers)
        self.encoder = nn.LSTM(
            input_size=hidden_layers, hidden_size=hidden_layers, batch_first=True, num_layers=1
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
        inputs = self.f(self.scale(inputs))
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
    def __init__(self, data, inputs, outputs):
        self.network = network(inputs, 256, inputs, outputs).to(device)
        self.network2 = network2(256, inputs, outputs).to(device)
        self.network3 = network3(256, inputs, outputs).to(device)
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

    def train(self, epoch, batch_size):
        
        epoch_size = int((len(self.data) * 1000) / batch_size)

        batched_states = []
        batched_next_actions = []

        for episode in self.data:
            
            states, next_actions = episode

            states = torch.stack(
                [torch.tensor(x, dtype=torch.float32, device=device) for x in states],
                dim=0,
            )

            next_actions = torch.stack(
                [torch.tensor(x, device=device) for x in next_actions], dim=0
            )

            split_states = states.split(10, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(10, 0)
            batched_next_actions.extend(split_actions)

        
        batched_states = torch.stack(batched_states[1:], dim=0)
        batched_next_actions = torch.stack(batched_next_actions[1:], dim=0)

        for e in range(epoch):
            attention = []
            lstm = []
            transformer = []
            for _ in range(epoch_size):

                indexes = np.random.choice(len(batched_states) - batch_size, batch_size, replace=False)
                batch_state = batched_states[indexes]
                batch_action = batched_next_actions[indexes][:, -1]

                prediction = self.network(batch_state)

                loss = self.loss_fn(prediction, batch_action)
                attention.append(loss.item())

                self.optim.zero_grad()
                # nn.utils.clip_grad_value_(self.network.parameters(), 1)
                loss.backward()
                self.optim.step()

                prediction2 = self.network2(batch_state)

                loss2 = self.loss_fn(prediction2, batch_action)
                lstm.append(loss2.item())

                self.optim2.zero_grad()
                # nn.utils.clip_grad_value_(self.network.parameters(), 1)
                loss2.backward()
                self.optim2.step()

                prediction3 = self.network3(batch_state)

                loss3 = self.loss_fn(prediction3, batch_action)
                transformer.append(loss3.item())

                self.optim3.zero_grad()
                # nn.utils.clip_grad_value_(self.network.parameters(), 1)
                loss3.backward()
                self.optim3.step()
            print("Epoch: ", e)
            print("Attention: ", sum(attention) / len(attention))
            print("LSTM: ", sum(lstm) / len(lstm))
            print("Transformer: ", sum(transformer) / len(transformer))
            print()

    def test(self, test_data, batch_size, epoch):
        print("now testing")
        batched_states = []
        batched_next_actions = []

        self.network.eval()
        self.network2.eval()
        self.network3.eval()

        epoch_size = int((len(self.data) * 1000) / batch_size)

        for episode in test_data:

            states, next_actions = episode
            states = torch.stack(
                [torch.tensor(x, dtype=torch.float32, device=device) for x in states],
                dim=0,
            )

            next_actions = torch.stack(
                [torch.tensor(x, device=device) for x in next_actions], dim=0
            )

            split_states = states.split(10, 0)
            batched_states.extend(split_states)

            split_actions = next_actions.split(10, 0)
            batched_next_actions.extend(split_actions)

        batched_states = torch.stack(batched_states[1:], dim=0)
        batched_next_actions = torch.stack(batched_next_actions[1:], dim=0)

        attention = []
        lstm = []
        transformer = []
        for _ in range(epoch_size * epoch):
            total = 0
            correct = 0
            total2 = 0
            correct2 = 0
            total3 = 0
            correct3 = 0
            indexes = np.random.choice(len(batched_states) - batch_size, batch_size, replace=False)
            batch_state = batched_states[indexes]
            batch_action = batched_next_actions[indexes][:, -1]

            prediction = self.network(batch_state)

            _, predicted = torch.max(prediction.data, 1)
            total += batch_action.size(0)
            correct += (predicted == batch_action).sum().item()
            attention.append(correct / total)

            prediction2 = self.network2(batch_state)
            _, predicted2 = torch.max(prediction2.data, 1)
            total2 += batch_action.size(0)
            correct2 += (predicted2 == batch_action).sum().item()
            lstm.append(correct2 / total2)

            prediction3 = self.network3(batch_state)
            _, predicted3 = torch.max(prediction3.data, 1)
            total3 += batch_action.size(0)
            correct3 += (predicted3 == batch_action).sum().item()
            transformer.append(correct3 / total3)

        print("Avg attention: ", sum(attention) / len(attention))
        print("Avg LSTM: ", sum(lstm) / len(lstm))
        print("Avg transformer: ", sum(transformer) / len(transformer))
        return (("attention", attention), ("lstm", lstm), ("transformer", transformer))
