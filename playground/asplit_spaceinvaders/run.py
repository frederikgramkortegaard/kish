""" 
Run the Atari Space Invaders game with an AttentionSplit based reinforcement learning agent. see model.py for the agent model. """

import sys
import os
import gymnasium as gym
import cv2
from skimage.util import crop
from skimage.color import rgb2gray

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# Meta
from kish.classes import Reinforcement
from kish.graphing import live_report_reinforcement_agent

from kish.runners import train_reinforcement_agent, iterative_train_reinforcement_agent
from kish.utils import save_reinforcement_agent_output

# Model Specifications
from model import Agent

if __name__ == "__main__":
    env = gym.make("SpaceInvadersNoFrameskip-v4", render_mode="rgb_array")
    space, _ = env.reset()
    state = cv2.resize(rgb2gray(crop(space, ((13, 13), (15, 25), (0, 0)))), (84, 84))
    env.metadata["render_fps"] = 1000
    agent = Agent(
        env=env,
        frames=5,
        n_step=5,
        batch_size=32,
        mem_size=15000,
        gamma=0.99,
        lr=0.003,
        weight_decay=1e-2,
        height=state.shape[0],
        width=state.shape[1],
        n_outputs=env.action_space.n,
        hidden_dim=576,
        epsilon=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.01,
    )

    live_report_reinforcement_agent(
        iterative_train_reinforcement_agent(
            agent, Reinforcement.TrainingInput(3e6, False)
        )
    )
