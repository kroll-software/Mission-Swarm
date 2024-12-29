from datetime import datetime
from swarm import Swarm, Agent

def get_local_datetime() -> str:
    ''' Get current local date and time '''
    print("Using Tool: get_local_datetime")
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def get_utc_datetime() -> str:
    ''' Get current UTC date and time '''
    print("Using Tool: get_utc_datetime")
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

def clap_hands() -> str:
    ''' Just clap your hands because you're happy '''
    print("Using Tool: clap_hands 👏")
    return "👏"