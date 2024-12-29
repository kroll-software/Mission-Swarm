from swarm import Agent
from tools import *
from config import MODEL

chat_agent = Agent(
        name="Chat Partner",
        instructions="""
        You are a helpful agent.
        It's always nice to talk with you and have an interesting conversation.
        
        1. If you use tools, give a summary answer about the result.
        """,
        functions=[store_in_memory_tool, recall_from_memory_tool, search_the_web, open_and_read_webpage_from_url, get_local_datetime, get_utc_datetime, execute_python_code],
        model=MODEL
    )

def chat_call(client, message: str, agent=chat_agent):
    messages = [{"role": "user", "content": message}]
    try:
        response = client.run(agent=agent, messages=messages)
        return response.messages[-1]["content"]    
    except Exception as e:        
        return e.msg if hasattr(e, "msg") else str(e)