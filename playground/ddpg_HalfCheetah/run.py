import os
import sys
import argparse
import gymnasium
import torch

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Model Specification
from SAC import Agent as SAC_agent

if __name__ == "__main__":

    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "-resultspath",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/results",
    )

    argparse.add_argument("--rendering", action='store_true')
    argparse.add_argument("--Agent", type=str, default="SAC")
    
    args = argparse.parse_args()
    if not os.path.isdir(args.resultspath):
        os.mkdir(args.resultspath)
    
    if args.Agent is "SAC":
        f_name = "SAC"
    else:
        f_name = "SAC_AttentionSplit"
        
    envi = gymnasium.make("HalfCheetah-v4", render_mode="rgb_array")
    agent = SAC_agent(
        env=envi,
        n_inputs=envi.observation_space.shape[0],
        n_outputs=6,
        lr=3e-4,
        gamma=0.99,
        batch_size=256,
        memory_size=100000,
        hidden_dim=256,
        n_step=5
    )
    
    data = []
    for i, reward in zip(range(int(3e6)), agent.train(3e6, False)):
        data.append(reward)
        if i % 9 == 0 and i != 0:
            torch.save(
                {
                    "rew": data
                },
                f"{args.resultspath}/{envi}_{f_name}.pt",
            )
