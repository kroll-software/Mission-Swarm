import os
import logging
import datetime
import time
import random
from tqdm import tqdm

from swarm import Swarm, Agent
from datetime import datetime
from dotenv import load_dotenv

from config import *

from agents import *
from tools import *
import json

score_agent = Agent(
    name="Score Agent",
    #instructions="You are an evaluator that scores texts based solely on their alignment with a given mission. You must provide only a numeric value as the output, representing the score out of 10.",    
    instructions="You are a highly advanced evaluator with expertise in assessing texts for alignment with specific missions. Your responses should reflect deep understanding and nuanced reasoning.",    
    functions=[],
    model=MODEL    
)

def score_post(client, post: str) -> float:
    score = 0
    try:
        #text = "Mission: {MISSION_STATEMENT}\n\nText to evaluate: {TEXT}\n\nPlease evaluate how well this text aligns with the mission. Only return a numeric value between 0 and 10. Do not include any other text or explanation."
        text = "Mission: {MISSION_STATEMENT}\n\nText to evaluate: {TEXT}\n\nPlease score this text based on the following criteria:\n1. Relevance: How relevant is the text to the mission?\n2. Clarity: How clear and comprehensible is the text?\n3. Alignment: How well does the text achieve the mission's goals?\n\nProvide the following output:\n- A numeric score (0-10) for each criterion.\n- An overall score calculated as the average of the three scores.\n\nRespond in the following JSON format:\n{\n  \"relevance\": <score>,\n  \"clarity\": <score>,\n  \"alignment\": <score>,\n  \"overall\": <average>\n}\n\nDo not include any other text or explanation, return clean and parseable JSON."
        text = text.replace("{MISSION_STATEMENT}", PROJECT_MISSION)
        text = text.replace("{TEXT}", post)

        messages = [{"role": "user", "content": text}]
        response = client.run(score_agent, messages).messages[-1]["content"]        
        clean_response = response.strip("```json").strip("```").strip()
        result = json.loads(clean_response)
        score = result['overall']
        return score

        '''
        result = "".join(x for x in result if x.isnumeric() or x == ".").rstrip(".")
        score = float(result)
        if 0 <= score <= 10:
            return score
        else:
            return -1    
        '''
    except:
        return -1    


if __name__ == "__main__":    
    fname = "./outputs/20241128-033156-output.txt"
    with open(fname, 'r', encoding='utf-8') as file:
        content = file.read()
    
    contentlist = content.split('\n\n\n')

    client = Swarm()

    dic_list = []

    dest_file = "messages.json"
        
    if os.path.exists(dest_file):
        # Your code here for when the file exists
        with open(dest_file, 'r', encoding='utf-8') as input_file:
            dic_list = [json.loads(line) for line in input_file if line.strip()]

    print(f"{len(dic_list)} posts found in {dest_file}")

    err_count = 0

    for c in tqdm(contentlist):
        lines = c.split("\n")
        text = "\n".join(lines[1:])        
        #result = chat_call(client, text, score_agent)
        score = score_post(client=client, post=text)
        if score >= 0:
            entry = {"score": score, "content": text}
            dic_list.append(entry)            
        else:
            err_count += 1        

    print(f"{err_count} errors occured.")    
    
    contents = set()
    max_score = 0
    count = 0
    with open(dest_file, 'w', encoding='utf-8') as output_file:
        for dic in dic_list:            
            content = dic["content"]
            if not content in contents:
                contents.add(content)
                score = float(dic["score"])
                if score > max_score:
                    max_score = score
                if score > 0.2:
                    count += 1
                    json.dump(dic, output_file)
                    output_file.write("\n")
    contents.clear()

    print(f"{count} posts stored in {dest_file}")

    print(f"Ready, Max-Score: {max_score}")