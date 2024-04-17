import sys
import os
import torch
import torch.nn as nn
import numpy as np

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


from ..playground.sarsa.Attention import Attention
