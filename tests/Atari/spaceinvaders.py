import os
import sys
import gym

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
    agent = Agent(frames=3, n_step=12, batch_size=128, mem_size=50000, gamma=0.99)

    live_report_reinforcement_agent(
        iterative_train_reinforcement_agent(
            agent, Reinforcement.TrainingInput(num_episodes=30000, render=True)
        )
    )
