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
    agent = Agent(3, 12, 32, 50000)

    live_report_reinforcement_agent(
        iterative_train_reinforcement_agent(
            agent, Reinforcement.TrainingInput(num_episodes=100, render=True)
        )
    )
