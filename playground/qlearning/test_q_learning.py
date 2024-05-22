import os
import sys
import argparse
import time
import gymnasium as gym
import torch

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
from playground.qlearning.qlearn import Agent as QLearningAgent
from Attention import agent as Agent2

if __name__ == "__main__":
    argparse = argparse.ArgumentParser()
    argparse.add_argument(
        "-resultspath",
        type=str,
        default=f"{os.path.dirname(os.path.abspath(__file__))}/results",
    )
    
    args = argparse.parse_args()
    if not os.path.isdir(args.resultspath):
        os.mkdir(args.resultspath)

    for _ in range(5):
        classic_control = ["Acrobot-v1", "MountainCar-v0", "CartPole-v1"]
        for env in classic_control:

            envi = gym.make(env, render_mode="rgb_array")

            agent = QLearningAgent(
                env=envi,
                n_inputs=envi.observation_space.shape[0],
                n_outputs=envi.action_space.n,
                lr=0.001,
                gamma=0.95,
                epsilon=1.0,
                epsilon_decay=0.9995,
                epsilon_min=0.01,
                batch_size=128,
                memory_size=10000,
            )

            for _ in agent.train(40000, False):
                pass
            
            print("Done training RL network")

            train_data = []
            for data in agent.test(20000, False):
                train_data.append(data)

            print("Done collecting train data")

            test_data = []
            for data in agent.test(5000, False):
                test_data.append(data)

            print("Done collecting test data")
            agent2 = Agent2(train_data, inputs=envi.observation_space.shape[0], outputs=envi.action_space.n)
            agent2.train(5, 128)
            results = agent2.test(test_data, 128, 2)
            attention, lstm, transformer = results
            
            # Save the results
            end_time = time.time()

            torch.save(
                {
                    attention[0] : attention[1],
                    lstm[0] : lstm[1],
                    transformer[0] : transformer[1]
                },
                f"{args.resultspath}/{env}_{end_time}.pt",
            )
