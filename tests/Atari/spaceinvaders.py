import os
import sys
import gymnasium as gym
import cv2
from skimage.util import crop
from skimage.color import rgb2gray
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.Atari.SpaceInvaders import Agent
from src.classes import Reinforcement
from src.graphing import (
    live_report_reinforcement_agent,
    save_reinforcement_agent_rewards_graph,
)

from src.runners import train_reinforcement_agent, iterative_train_reinforcement_agent
from src.utils import save_reinforcement_agent_output

if __name__ == "__main__":
    env=gym.make("SpaceInvadersNoFrameskip-v4", render_mode="human")
    space, _ = env.reset()
    state = cv2.resize(
            rgb2gray(crop(space, ((13, 13), (15, 25), (0, 0)))), (84, 84)
        )
    env.metadata['render_fps'] = 1000
    agent = Agent(env=env, 
                  frames=10, 
                  n_step=10, 
                  batch_size=32, 
                  mem_size=10000, 
                  gamma=0.99, 
                  lr=3e-4, 
                  weight_decay=1e-2, 
                  height=state.shape[0], 
                  width=state.shape[1],
                  n_outputs=env.action_space.n,
                  hidden_dim=576)


    live_report_reinforcement_agent(iterative_train_reinforcement_agent(agent, Reinforcement.TrainingInput(3e6, True)))
