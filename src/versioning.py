import time
from disk import get_agent_name


def verify_agent(
    agent: object,
):
    """Assert that the model has the correct fields"""
    assert hasattr(agent, "train")
    assert hasattr(agent, "name")
    assert hasattr(agent, "description")


def name_agent(agent) -> str:
    """Name an agent"""
    agent.hashname = get_agent_name(agent)
    agent.namedtime = time.time()
    return agent.hashname
