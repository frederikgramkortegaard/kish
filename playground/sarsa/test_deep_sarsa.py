import os
import sys
import gym

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)


# Meta
from kish.classes import Reinforcement
from kish.utils import save_reinforcement_agent_output

# Model Specification
from sarsa import Agent as DeepSARSA
from Attention import agent as Agent2


if __name__ == "__main__":

    envi = gym.make("CartPole-v1")
    agent = DeepSARSA(
        env=envi,
        n_inputs=envi.observation_space.shape[0],
        n_outputs=2,
        lr=0.003,
        gamma=0.9,
        epsilon=1.0,
        epsilon_decay=0.9995,
        epsilon_min=0.01,
        batch_size=32,
        memory_size=10000,
    )

    for i in agent.train(40, True):
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
