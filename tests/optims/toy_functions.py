import numpy as np
import torch.nn as nn
import os 
import sys
import torch
import torch.distributions as dist
import math
import numpy as np
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
from models.Atari.Optimizer import NAdam, SGDNorm, RNAdamWP, SGD, NSGD, LevenbergMarquardt, SG
device = torch.device("cuda")

# Initializes weights with orthonormal weights row-wise
def Orthonorm_weight(weight):
    ones = torch.ones_like(weight).data.normal_(0, math.sqrt(2 / (weight.numel()))).t() # We choose Microsoft init as our choice of basis for the vector-space

    for i in range(1, int(ones.shape[0])):
        projection_neg_sum = torch.zeros_like(ones[i, :])
        for j in range(i):
            projection_neg_sum.data.add_((((ones[i, :].t() @ ones[j, :]) / (ones[j, :].t() @ ones[j, :])) * ones[j, :]))
        ones[i, :].data.sub_(projection_neg_sum)
        
    ones /= torch.sqrt((ones ** 2).sum(-1, keepdim=True))

    return ones.t() # Return Orthonormal basis

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()

        self.fc1 = nn.Linear(1, 256)
        self.fc1.weight.data.normal_(0, math.sqrt(2/self.fc1.weight.numel()))
        
        self.ln = nn.LayerNorm(256)

        self.fc4 = nn.Linear(256, 1)
        self.fc4.weight.data.normal_(0, math.sqrt(2/self.fc4.weight.numel()))

        self.f = nn.SELU()

    def forward(self, inputs):
        x = self.f(self.ln(self.fc1(inputs)))
        return self.fc4(x)
    
if __name__ == "__main__":

    model = Network().to(device)
    optim = SGD(model.parameters(), lr=0.1, momentum=0.9)

    for i in range(15000):
        data = torch.zeros((4096, 1), device=device).data.normal_(0, 1)
        predict = model(data)
        loss = (predict - data.pow(4).add(2*data.pow(2)).sub(1000*data)).pow(2).mean()

        optim.zero_grad()
        loss.backward()
        optim.step()
        
        with torch.no_grad():
            data = torch.zeros((256, 1), device=device).data.normal_(0, 1)
            model.eval()
            eval = (model(data) - data.pow(4).add(2*data.pow(2)).sub(1000*data)).pow(2).mean()
            model.train()
            print(eval)
            print()
        #input()