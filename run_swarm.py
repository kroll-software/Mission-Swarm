import os
import shutil
import logging
import random
import time
from datetime import datetime
from typing import List, Dict
import json

# imports from this project
from config import *
from agents import *
from tools import *
from chatmemory import *

settings_dir = ".settings"
history_file = os.path.join(settings_dir, "history.json")

def store_history(history: List):
    try:
        with open(history_file, 'w', encoding='utf-8') as output_file:
            for dic in history:
                json.dump(dic, output_file)
                output_file.write("\n")
    except Exception as e:
        print(f"Error storing chat history: {e.msg if hasattr(e, "msg") else str(e)}")

def load_history():
    try:
        print(f"Loading chat history from {history_file} ...")
        if os.path.exists(history_file):
            with open(history_file, 'r', encoding='utf-8') as file:            
                for line in file:
                    if line.strip():
                        message = json.loads(line)
                        put_chat_message(message)
    except Exception as e:
        print(f"Error loading chat history: {e.msg if hasattr(e, "msg") else str(e)}")

sleep_time = 5

def score2stars(score: float) -> str:    
    return "⭐" * round(score) 

def is_empty_message(msg: str):
    if not msg:
        return True    
    return not msg.strip(" \n")

def main_chat_loop():    
    load_history()
    history = get_chat_history()

    if len(history) == 0:
        topic = PROJECT_MISSION    
        put_chat_message({"role": "user", "content": topic})        
        file_logger.info(msg="\n***** TOPIC *****\n")
        file_logger.info(msg=topic + "\n\n")
        
    i = 0
    emptycount = 0
    while True:        
        i += 1
        random_agent = random.choice(agent_list())

        try:
            history = get_chat_history()            
            token_count = get_chat_history_token_count()
            print(f"History Length: {len(history)}, Tokens: {token_count}")
            store_history(history)

            if len(history) == 0 or history[-1]["role"] != "user":
                if emptycount > 10:
                    emptycount = 0
                    history.append({"role": "user", "content": "Proceed with your mission. Try another way."})                
                    print("Proceed with your mission. Try another way.")
                else:
                    if i % 7 == 0:
                        random_agent = project_manager
                        history.append({"role": "user", "content": "Project manager, your turn."})
                    else:
                        history.append({"role": "user", "content": "Proceed, go deep."})                        

            start_time = time.time()
            response = CLIENT.run(agent=random_agent, messages=history)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_minutes = int(elapsed_time // 60)
            elapsed_seconds = int(elapsed_time % 60)
            str_elapsed = f"{elapsed_minutes}:{elapsed_seconds:02}"

            post = response.messages[-1]["content"]

            if not is_empty_message(post):
                score = score_post(CLIENT, post)
                stars = score2stars(score)
                timestr = datetime.now().strftime("%d.%m.%Y %H:%M")                        
                post = f"{response.agent.name.upper()} ({timestr}) | Score: {stars}\n{post}"
                file_logger.info(msg=post + "\n\n")
                message = response.messages[-1]
                message["score"] = int(score)
                put_chat_message(message)
                print(f"response OK, Score: {score:.1f}, Time: {str_elapsed}")
                emptycount = 0
                time.sleep(sleep_time)
            else:
                emptycount += 1
                print(f"empty response from agent ({emptycount})")
            
        except Exception as e:
            emptycount += 1
            print(e.msg if hasattr(e, "msg") else str(e))        

def setup_filelogger(logfile_name: str = None) -> str:
    global file_logger
    file_logger = logging.getLogger(__name__)
    file_logger.setLevel(logging.INFO)    

    if not logfile_name:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        logfile_name = os.path.join("outputs", timestr + "-output.md")

    print(f"Logging chat to file: {logfile_name}")
    file_handler = logging.FileHandler(logfile_name)
    file_handler.setLevel(logging.INFO)
    #formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')    
    #file_handler.setFormatter(formatter)
    file_logger.addHandler(file_handler)
    return logfile_name

def delete_all_project_data():
    # delete the chat-history
    if os.path.exists(history_file):
        print(f"deleting history {history_file} ...")
        try:
            os.remove(history_file)
        except Exception as e:
            print(e.msg if hasattr(e, "msg") else str(e))

    # delete stored agents
    agents_file = "agents.pkl"
    if os.path.exists(agents_file):
        print(f"deleting agents {agents_file} ...")
        try:
            os.remove(agents_file)
        except Exception as e:
            print(e.msg if hasattr(e, "msg") else str(e))


    # delete the chroma store when starting a fresh run.
    chroma_path = ".chroma"
    if os.path.exists(chroma_path):
        print(f"deleting chroma store {chroma_path} ...")
        try:
            shutil.rmtree(chroma_path)
        except Exception as e:
            print(e.msg if hasattr(e, "msg") else str(e))    

def makedir(dir: str):
    if not os.path.exists(dir):
        print(f"Creating directory '{dir}' ...")
        os.mkdir(dir)        


if __name__ == "__main__":

    # create some dirs    
    makedir(".settings")
    makedir("outputs")
    makedir("workspace")
    
    # continue last run or start from scratch
    start_from_scratch = True    
    last_run_file = os.path.join(settings_dir, "last_run.json")
    logfile_name = ""
    if os.path.exists(last_run_file):       
        with open(last_run_file, "r") as f:            
            last_run_json = json.load(f)
        
        mission = last_run_json["mission"]
        if mission == PROJECT_MISSION:            
            logfile_name = last_run_json["logfile"]
            start_from_scratch = False

    logfile_name = setup_filelogger(logfile_name)

    # store current settings    
    last_run_data = {
        "mission": PROJECT_MISSION,
        "logfile": logfile_name,        
        "date": time.strftime("%Y-%m-%d %H:%M:%S")
    }

    with open(last_run_file, "w") as outfile: 
        json.dump(last_run_data, outfile, indent=4)

    # delete all prior data when starting from scratch
    if start_from_scratch:        
        delete_all_project_data()    

    if start_from_scratch:
        file_logger.info(msg=f"LLM: {os.path.basename(MODEL)}")
        timestr = datetime.now().strftime("%m/%d/%Y, %H:%M:%S")
        file_logger.info(msg=f"Date: {timestr}")
    
    init_long_term_memory()
    init_chat_memory(token_limit=CONTEXT_LENGTH // 2)
        
    agents_file = os.path.join(settings_dir, "agents.pkl")
    init_agents(50, agents_filename=agents_file)
    
    # run the main loop endlessly
    main_chat_loop()
        