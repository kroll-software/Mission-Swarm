from dotenv import load_dotenv
from swarm import Swarm

load_dotenv()

PROJECT_MISSION = """
THE MISSION: Develop an automated mathematical proof for the Riemann hypothesis.
"""

# Changing the PROJECT_MISSION will delete all prior data and start from scratch 
# You can also delete ".settings" to start from scratch

# our model with 48k context, see README.md how to create it
MODEL = "qwen2.5-coder-14b-48k:latest"

# a local path or a Huggingface path for a Transformers Tokenizer
# should correspond with the model above
TOKENIZER_MODEL_PATH = "Qwen/Qwen2.5-Coder-32B-Instruct"

# should corresponds with the model above
# this setting does NOT change the context size.
CONTEXT_LENGTH = 49152

CLIENT = Swarm()
