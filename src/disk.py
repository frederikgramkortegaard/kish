import os
import pickle
import time
import logging
import sys

# SHA256
import hashlib


def get_agent_name(agent):
    """Get the name of an agent"""
    hashname = hashlib.sha256(
        agent.__class__.__name__.encode("utf-8") + str(time.time()).encode("utf-8")
    ).hexdigest()
    logging.info(f"Created agent hashname: {hashname}")
    return hashname


def dump_results(config, agent_hashname, results):
    with open(
        os.path.join(
            config["DEFAULT"]["results_dump_path"],
            f"{agent_hashname}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(results, f)


def dump_agent(config, agent):
    with open(
        os.path.join(
            config["DEFAULT"]["agent_dump_path"],
            f"{agent.hashname}.pkl",
        ),
        "wb",
    ) as f:
        pickle.dump(agent, f)


def dump_graph(config, agent_hashname, plt):
    plt.savefig(f'{config["DEFAULT"]["path_to_graphs"]}/{agent_hashname}.png')
