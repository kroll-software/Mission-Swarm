from dotenv import load_dotenv
from swarm import Swarm

load_dotenv()

PROJECT_MISSION = """
THE MISSION: Develop an automated mathematical proof for the Riemann hypothesis.
"""

'''
Remarks:
1. Changing the PROJECT_MISSION will delete all prior data and start from scratch 
2. You can also delete "./last_run.json" to start from scratch
'''

#MODEL = "qwen2.5-coder:32b"
#MODEL = "qwen2.5-coder-longctx"
#MODEL = "qwen2.5-coder:14b"
MODEL = "qwen2.5-coder-14b-48k:latest"

# a local path or a Huggingface path for a Transformers Tokenizer
TOKENIZER_MODEL_PATH = "Qwen/Qwen2.5-Coder-32B-Instruct"

#CONTEXT_LENGTH = 32768  # 35900 geht auch
CONTEXT_LENGTH = 49152
#CONTEXT_LENGTH = 8192  # 35900 geht auch
#CONTEXT_LENGTH = 16384  # 35900 geht auch
#MODEL = "qwq"
#PROJECT_MISSION = "THE MISSION: Develop a mathematical proof for the Riemann Hypothesis."

CLIENT = Swarm()
#SCORE_CLIENT = Swarm(temperature=0)
