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
from SAC_Attention import Agent as SAC_Attention_agent

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
    

    import time
    start_time = time.time()
        
    envi = gymnasium.make("Walker2d-v4", render_mode="rgb_array")
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


    print("Starting SAC (No AttentionSplit)")
    
    data = []
    for i, avg_rewards in zip(range(int(1e6)), agent.train(1e6, False)):
        if i % 5 == 0 and i != 0:
            print(i, "Adding avg_reward")
            data.append(avg_rewards)
            torch.save(
                {
                    "rew": data
                },
                f"{args.resultspath}/SAC_{start_time}.pt",
            )

    
    print("Starting SAC (With AttentionSplit)")

    envi = gymnasium.make("Walker2d-v4", render_mode="rgb_array")
    agent = SAC_Attention_agent(
        env=envi,
        frames=5,
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
    for i, avg_rewards in zip(range(int(1e6)), agent.train(1e6, False)):
        if i % 5 == 0 and i != 0:
            print(i, "Adding avg_reward")
            data.append(avg_rewards)
            torch.save(
                {
                    "rew": data
                },
                f"{args.resultspath}/SAC_Attention_{start_time}.pt",
            )