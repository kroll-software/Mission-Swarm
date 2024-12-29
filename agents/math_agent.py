import ast
from swarm import Agent
from tools import execute_python_code
from config import MODEL

math_agent = Agent(
    name="Math Genie",
    instructions="""
    You are a helpful genius math agent.
    1. Whenever you use tools, you extract the results and give a clear answer.
    2. Whenever you write python-code, it is clean and directly executable.    
    """,
    functions=[execute_python_code],
    model=MODEL
)

def math_agent_call(client, message: str):
    agent = math_agent
    messages = [{"role": "user", "content": message}]
    response = client.run(agent=agent, messages=messages)

    return response.messages[-1]["content"]    