import torch
import torch.nn as nn
import math

device = torch.device('cuda')

class AttentionSplit(nn.Module):
    def __init__(self, inputs, hidden_size, n_heads):
        super(AttentionSplit, self).__init__()
        self.inputs = inputs
        self.hidden_size = hidden_size
        self.n_heads = n_heads

        nc = self.n_heads * int(self.hidden_size / self.n_heads)
        self.cells = nn.Parameter(torch.ones((self.n_heads, int(self.hidden_size / self.n_heads))))
        nn.init.normal_(self.cells, 0, math.sqrt(2/nc))

        na = self.n_heads * int(int(self.hidden_size * self.n_heads) / self.inputs)
        self.cell_attention = nn.Parameter(torch.ones((self.n_heads, int(int(self.hidden_size * self.n_heads) / self.hidden_size))))
        nn.init.normal_(self.cell_attention, 0, math.sqrt(2/na))

        self.head_weight = nn.Parameter(torch.ones((self.n_heads, int(int(self.hidden_size * self.n_heads) / self.inputs))))
        nn.init.normal_(self.head_weight, 0, math.sqrt(2/na))
        
        self.last_conv = nn.Conv2d(1, self.n_heads, kernel_size=int(self.n_heads-1), padding=int((self.n_heads-1)/2))
        nn.init.normal_(self.last_conv.weight, 0, math.sqrt(2/(self.n_heads -1)))

        self.dropout = nn.Dropout(0.1)

        self.layer_norm = nn.LayerNorm(self.hidden_size)
        self.first_norm = nn.LayerNorm(int(self.hidden_size / self.n_heads))
        self.cell_norm = nn.LayerNorm(int(self.hidden_size / self.n_heads))
        self.cell_attention_norm = nn.LayerNorm(int(self.hidden_size / self.n_heads))

        self.f = nn.GELU()
        self.g = nn.SELU()

        if (self.inputs % self.n_heads) != 0 or (self.hidden_size % self.n_heads) != 0 or (self.inputs % self.n_heads) == 1 or ((self.hidden_size * self.n_heads) / self.inputs) % 1 != 0:
            print("Error: input must be divisable by n_heads")
            raise AttributeError

        self.heads = nn.ModuleList()

        self.to(device)

    def forward(self, inputs, hidden=None):
        if inputs.dim() < 3:
            inputs = inputs.unsqueeze(1)

        bs, ls, _ = inputs.shape
        current_input = inputs.view(inputs.shape[0], inputs.shape[1], self.n_heads, -1).view(-1, self.n_heads)

        # Calculate the outputs for all timesteps in a matrix multiplication
        pure_outputs = self.f(self.first_norm((current_input @ self.head_weight).view(bs, ls, self.n_heads, -1)))

        if hidden is not None:
            hidden = self.f(hidden.view(-1, self.n_heads) @ self.cell_attention)
            hidden = self.cell_attention_norm(hidden.view(bs, ls, self.n_heads, int(self.hidden_size / self.n_heads)))

        temporals = torch.zeros(bs, self.n_heads, int(self.hidden_size / self.n_heads)).to(device)
        output_list = []
        context_window = []

        if hidden is None:
            for i in range(inputs.shape[1]):

                context_outputs = torch.einsum('nhi, bhj->bhj', self.cells.unsqueeze(0), self.dropout(temporals))
                context_outputs = self.dropout(self.f(context_outputs))

                combined_outputs = pure_outputs[:, i, :] + context_outputs
                combined_output = self.f(self.cell_norm(combined_outputs))
                output_list.append(combined_output.unsqueeze(1))

                temporals = self.g(combined_outputs)
                context_window.append(temporals)
        else:
            for i in range(inputs.shape[1]):

                context_outputs = torch.einsum('nhi, bhj->bhj', self.cells.unsqueeze(0), self.dropout(temporals + hidden[:, i, :, :]))
                context_outputs = self.dropout(self.f(context_outputs))

                combined_outputs = pure_outputs[:, i, :] + context_outputs
                combined_output = self.f(self.cell_norm(combined_outputs))
                output_list.append(combined_output.unsqueeze(1))

                temporals = self.g(combined_outputs)
                context_window.append(temporals)

        outputs = torch.cat(output_list, dim=1).view(bs, 1, ls, self.hidden_size)
        conv_outputs = self.f(self.last_conv(outputs))

        for i in range(conv_outputs.shape[1]):
            if i == 0:
                continue
            conv_outputs[:, i, :, :] = self.g(conv_outputs[:, i, :, :] - conv_outputs[:, i-1, :, :]) # Suppose these activations are results of Gaussian kernels, we then take the DAG across heads
        outputs = self.f(self.layer_norm(conv_outputs[:, -1, :, :].view(bs, ls, self.hidden_size)))

        return outputs, torch.stack(context_window, dim=1)