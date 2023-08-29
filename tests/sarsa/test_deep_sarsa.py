import os
import sys
import gym

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from models.sarsa.sarsa import Agent as DeepSARSA
from src.classes import Reinforcement
from src.graphing import (
    live_report_reinforcement_agent,
    save_reinforcement_agent_rewards_graph,
)
from src.runners import train_reinforcement_agent, iterative_train_reinforcement_agent
from src.utils import save_reinforcement_agent_output

if __name__ == "__main__":

    agent = DeepSARSA(
        env=gym.make("CartPole-v1"),
        n_inputs=4,
        n_outputs=2,
        lr=0.01,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=10000,
    )

    live_report_reinforcement_agent(
        iterative_train_reinforcement_agent(
            agent, Reinforcement.TrainingInput(num_episodes=100, render=False)
        )
    )
