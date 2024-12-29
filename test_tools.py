from config import *
from agents import *
from tools import *
from chatmemory import *

if __name__ == "__main__":
    
    agent = create_person_agent()

    if False:
        # web-search test 
        q = "Summarize George Washington's wikipedia page"
        print(q + "\n")        
        post = chat_call(CLIENT, q, agent)
        print(post + "\n")

    # long-term memory test
    init_long_term_memory()

    q = "Please remember: My mother was born in 1940"
    print(q + "\n")
    post = chat_call(CLIENT, q, agent)    
    print(post + "\n")

    q = "When was my mother born? recall from memory."
    print(q + "\n")
    post = chat_call(CLIENT, q, agent)
    print(post + "\n")

    # Python evaluation test
    q = "What is the sum of the first 20 fibonacci numbers?"
    print(q + "\n")
    response = math_agent_call(CLIENT, q)
    print(response + "\n")

    print("Ready.")