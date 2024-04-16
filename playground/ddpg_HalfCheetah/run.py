import os
import sys
import gymnasium

sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

# Meta
from kish.classes import Reinforcement
from kish.graphing import (
    live_report_reinforcement_agent,
    save_reinforcement_agent_rewards_graph,
)
from kish.runners import train_reinforcement_agent, iterative_train_reinforcement_agent
from kish.utils import save_reinforcement_agent_output

# Model Specification
from SAC import Agent as SAC_agent

if __name__ == "__main__":

    envi = gymnasium.make("HalfCheetah-v4", render_mode="rgb_array")
    agent = SAC_agent(
        env=envi,
        n_inputs=envi.observation_space.shape[0],
        n_outputs=6,
        lr=0.01,
        gamma=0.99,
        batch_size=256,
        memory_size=100000,
    )

    live_report_reinforcement_agent(
        iterative_train_reinforcement_agent(
            agent, Reinforcement.TrainingInput(3e6, False)
        )
    )
