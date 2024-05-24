""" 
Run the Atari Space Invaders game with an AttentionSplit based reinforcement learning agent. see model.py for the agent model. """

import sys
import os
import gymnasium as gym
import cv2
from skimage.util import crop
from skimage.color import rgb2gray
import torch
import argparse

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# Model Specifications
from model import Agent

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
    
    env = gym.make("SpaceInvadersDeterministic-v4", render_mode="rgb_array")
    env.metadata["render_fps"] = 10000
    space, _ = env.reset()
    state = cv2.resize(rgb2gray(crop(space, ((13, 13), (15, 25), (0, 0)))), (84, 84))
    wrap = gym.wrappers.HumanRendering(env)
    
    agent = Agent(
        env=wrap,
        frames=5,
        n_step=5,
        batch_size=32,
        mem_size=30000,
        gamma=0.999,
        lr=3e-3,
        weight_decay=1e-2,
        height=state.shape[0],
        width=state.shape[1],
        n_outputs=env.action_space.n,
        hidden_dim=256,
        epsilon=1.0,
        epsilon_decay=0.99985,
        epsilon_min=0.01,
    )
    f_name = "AttentionSplit_SpaceInvaders"
    data = []
    for i, reward in zip(range(int(3e6)), agent.train(3e6, False)):
        data.append(reward)
        if i % 9 == 0 and i != 0:
            torch.save(
                {
                    "rew": data
                },
                f"{args.resultspath}/{wrap}_{f_name}.pt",
            )
