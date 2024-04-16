import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Variational Bayesian linear layer implementation
class VBLinear(nn.Module):

    def __init__(self, in_dim, out_dim):
        super(VBLinear, self).__init__()

        self.mu_w = nn.Parameter(torch.ones(out_dim, in_dim))
        self.varsig = nn.Parameter(torch.ones(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))

        self.mu_w.data.zero_().normal_(0, 0.1)  # Narrow normal dist
        self.varsig.data.zero_().normal_(-9, 0.001)  # var init via Louizos
        self.bias.data.zero_()
        self.prior_prec = 10

        self.to(device)

    def KL(self):

        logsig2_w = self.varsig.clamp(-11, 11)
        kl = (
            0.5
            * (
                self.prior_prec * (self.mu_w.pow(2) + logsig2_w.exp())
                - logsig2_w
                - 1
                - np.log(self.prior_prec)
            ).mean()
        )
        return kl

    def forward(self, input):
        if self.training is False:
            return F.linear(input, self.mu_w, self.bias)
        else:
            mu_out = F.linear(input, self.mu_w, self.bias)
            logsig2_w = self.varsig.clamp(-11, 11)
            s2_w = logsig2_w.exp()
            var_out = F.linear(input.pow(2), s2_w) + 1e-8
            return mu_out + var_out.sqrt() * torch.randn_like(mu_out)
