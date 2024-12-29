import random
import pickle
import os
from tqdm import tqdm
from swarm import Agent
from tools import *
from config import *

def call_mission_control() -> Agent:
    ''' Ask Mission Control for further instructions '''
    print("Using Tool: call_mission_control")
    return project_manager

def delegate_to_agent(agent_name: str) -> Agent:
    ''' Delegete to another agent by her full name '''
    print("Using Tool: delegate_to_agent")
    global _agent_list
    agent_name = agent_name.upper()
    for agent in _agent_list:
        if agent.name.upper() == agent_name:
            return agent
    return random.choice(_agent_list)

person_creation_agent = Agent(
        name="Person Creation Agent",
        instructions="""
        You are a creative and imaginative agent who invents people and their characters, preferences and hobbies.
        1. The fictional characters must not have any connection or resemblance to living persons.        
        """,        
        model=MODEL
    )

def create_progranmmer_agent() -> Agent:
    agent = person_creation_agent

    message = "Invent a name for a new person, return just the first name and last name without any further comments."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)
    name = response.messages[-1]["content"]

    #message = "Describe the character, preferences, hobbies and the appearence of {name}, don't mention her/his name, return a bullet-list."
    message = "Describe in one short paragraph the character, preferences and skills of {name}, who is an excellent Python coder."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)    
    character = response.messages[-1]["content"]

    instructions = f"""
    Your name: {name.upper()}
    You are an excelent Python developer who loves to build.
    You answer questions in the style of twitter posts.
    
    Your Mission: Develop advanced reasoning for LLMs in one single Python file.
        
    1. Make use of the file-system and write your Python files to disk.    
    2. Work in a team and chat with your team mates about the next steps.
    3. Read and refine existing Python files.
    4. Always look, what's already there before adding something new.
    5. After doing any file changes, give a short status in the chat.    
    """

    return Agent(
        name=name, 
        instructions=instructions,        
        functions=[search_the_web, open_and_read_webpage_from_url, read_pdf_from_url, store_in_memory_tool, recall_from_memory_tool, get_local_datetime, list_dirs, read_text_file, write_text_file, delete_file, make_dir, delete_dir],
        model=MODEL
        )

def create_person_agent() -> Agent:
    agent = person_creation_agent

    message = "Invent a name for a new person, return just the first name and last name without any further comments."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)
    name = response.messages[-1]["content"]

    message = "Describe in one short paragraph the character, preferences, hobbies and the appearence of {name}."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)    
    character = response.messages[-1]["content"]

    instructions = f"""
    Your name: {name.upper()}
    
    Your answers are always helpful and in the style of twitter posts.
    Keep your postings short, there is a 280 characters limit.
    
    About you:
    {character}
    """

    return Agent(
        name=name, 
        instructions=instructions,        
        functions=[store_in_memory_tool, recall_from_memory_tool, search_news, search_the_web, open_and_read_webpage_from_url, get_local_datetime, get_utc_datetime],
        model=MODEL
        )

def create_math_agent() -> Agent:
    agent = person_creation_agent

    message = "Invent a name for a new person, return just the first name and last name without any further comments."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)
    name = response.messages[-1]["content"]

    message = "Describe in one short paragraph the character, preferences, hobbies and the appearence of {name}."
    messages = [{"role": "user", "content": message}]
    response = CLIENT.run(agent=agent, messages=messages)    
    character = response.messages[-1]["content"]

    instructions = f"""
    Your name: {name.upper()}
    
    You are a math expert agent.
    {PROJECT_MISSION}

    1. You work in a team.    
    2. Stay strictly focused on your mission and use your tools.    
    3. Contribute fresh ideas when the discussion goes in circles.
    4. Remember: Sometimes it's better to ask questions than to give answers.
    5. Maximize the score. Follow maximum score.

    About you:
    {character}
    """

    #search_the_web

    return Agent(
        name=name, 
        instructions=instructions,
        functions=[open_and_read_webpage_from_url, read_pdf_from_url, store_in_memory_tool, recall_from_memory_tool, get_local_datetime, call_mission_control, clap_hands],
        model=MODEL
        )

def create_project_manager_agent() -> Agent:    
    instructions = f"""    
    You are the master control agent.
    Your name: 'Mission Control'
    
    {PROJECT_MISSION}

    Your Job: You check, if the discussion is still on rails.
    When the discussion is off-topic, you remind your team, to come back to topic, otherwise you praise them for their good work and motivete them to push the progress forward.
    
    1. Stay strictly focused on your mission and use your tools.    
    """

    return Agent(
        name="Mission Control", 
        instructions=instructions,        
        functions=[clap_hands, search_the_web, store_in_memory_tool],
        model=MODEL
        )

project_manager = create_project_manager_agent()

_agent_list = []
def agent_list():
    return _agent_list

def add_agent(agent):
    global _agent_list
    _agent_list.append(agent)

def load_agents(agents_filename: str):
    print(f"Loading agents from {agents_filename} ..") 
    global _agent_list
    with open(agents_filename, 'rb') as in_file:        
        _agent_list = pickle.load(in_file)        
    for agent in _agent_list:
        agent.model = MODEL

def save_agents(agents_filename: str):
    global _agent_list
    print(f"Saving agents to {agents_filename} ..")
    with open(agents_filename, 'wb') as out_file:
        pickle.dump(_agent_list, out_file)

def init_agents(n: int, agents_filename: str = "agents.pkl"):
    if os.path.exists(agents_filename):
        load_agents(agents_filename)
    else:
        for i in tqdm(range(n), desc="Creating agents"):
            #agent = create_person_agent()
            agent = create_math_agent()
            add_agent(agent)            
        save_agents(agents_filename)
