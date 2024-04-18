import os
import sys
import gymnasium as gym

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Meta
from kish.classes import Reinforcement, Network
from kish.utils import save_reinforcement_agent_output
from kish.runners import (
    iterative_run_reinforcement_agent,
    iterative_train_reinforcement_agent,
    train_network,
)
from kish.graphing import live_report_reinforcement_agent

# Model Specification
from sarsa import Agent as DeepSARSA
from Attention import agent as Agent2

if __name__ == "__main__":

    envi = gym.make("Acrobot-v1")
    agent = DeepSARSA(
        env=envi,
        n_inputs=envi.observation_space.shape[0],
        n_outputs=envi.action_space.n,
        lr=0.003,
        gamma=0.95,
        epsilon=1.0,
        epsilon_decay=0.99995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=5000,
    )

    for i in agent.train(40, False):
        l = i

    k = []
    for i in agent.test(60, False):
        k.append(i)

    p = []
    for i in agent.test(7, False):
        p.append(i)

    agent2 = Agent2(k)
    agent2.train()
    agent2.test(p)
