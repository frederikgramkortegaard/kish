from matplotlib import pyplot as plt
import torch


data = torch.load("SAC_Attention_1716978383.3201127.pt")
data2 = torch.load("SAC_1716978383.3201127.pt")

plt.plot(data["rew"][:294], label="Attention")
plt.plot(data2["rew"][:294], label="No Attention")

# x label is in 5000
plt.xlabel("# of 5000 timesteps")

plt.legend()
plt.show()
