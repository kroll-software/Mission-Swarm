#
# This code is inspired by the llama_index library under MIT License
# with many thanks and great respect for their work
#

from copy import deepcopy, copy
from typing import List, Dict, Tuple, Any, Optional, Callable
from config import *
from swarm import Agent
from tools import clap_hands
from transformers import AutoTokenizer

DEFAULT_TOKEN_LIMIT_RATIO = 0.75
DEFAULT_TOKEN_LIMIT = 2000
SUMMARIZE_PROMPT = "The following is a conversation between the user and assistant. Write a concise summary about the contents of this conversation."


# Load the Qwen tokenizer
tokenizer = None

def tokenize_string(text: str) -> List:
    global tokenizer

    if not text:
        return []

    # Tokenize the input text
    tokens = tokenizer.encode(text, add_special_tokens=True)    

    # Return the token count
    return tokens


def create_summary_agent() -> Agent:
    return Agent(name="Summary Agent",
        instructions="""
        You are a Summarization agent.        
        You write comprehensive and accurate summaries of provided content.
        """,
        functions=[],
        model=MODEL,
    )

def summarize_fn(messages: List[Dict]) -> str:    
    global summary_agent
    all_message = "\n\n".join([x["content"] for x in messages])    
    message = [{"role": "user", "content": all_message + "\n\nSummarize all text above."}]
    result = CLIENT.run(agent=summary_agent, messages=message)
    return result.messages[-1]["content"]

#********************* Classes **************************

# TODO: Add option for last N user/assistant history interactions instead of token limit
class ChatSummaryMemoryBuffer():
    def __init__(self, token_limit: int, count_initial_tokens: bool = False, summarize_fn: Callable[[List[Dict]], str] = None, tokenizer_fn: Callable[[str], List] = None, chat_history = None):
        self.token_limit = token_limit
        self.count_initial_tokens = count_initial_tokens        
        self.summarize_fn = summarize_fn
        self.tokenizer_fn = tokenizer_fn
        #self.chat_store: SerializeAsAny[BaseChatStore] = Field(default_factory=SimpleChatStore)
        self.chat_store = []
        if chat_history is not None:
            self.chat_store.extend(chat_history)

        #self.chat_store_key = DEFAULT_CHAT_STORE_KEY
        self._token_count = 0

    def get(self, input: Optional[str] = None, initial_token_count: int = 0, **kwargs: Any) -> List[Dict]:
        """Get chat history."""
        chat_history = self.get_all()
        if len(chat_history) == 0:
            self._token_count = 0
            return []

        # Give the user the choice whether to count the system prompt or not
        if self.count_initial_tokens:
            if initial_token_count > self.token_limit:
                raise ValueError("Initial token count exceeds token limit")
            self._token_count = initial_token_count

        (chat_history_full_text, chat_history_to_be_summarized) = self._split_messages_summary_or_full_text(chat_history)

        if len(chat_history_to_be_summarized) == 0:
            # Simply remove the message that don't fit the buffer anymore
            updated_history = chat_history_full_text
        else:                        
            updated_history = [                
                self._summarize_oldest_chat_history(chat_history_to_be_summarized),
                *chat_history_full_text,
            ]

        self.reset()
        self._token_count = 0
        self.set(updated_history)

        return updated_history

    def get_all(self) -> List[Dict]:
        """Get all chat history."""
        return self.chat_store.copy()

    def put(self, message: Dict) -> None:
        """Put chat history."""        
        self.chat_store.append(message)
        self._token_count += self._token_count_for_messages([message])

    def set(self, messages: List[Dict]) -> None:
        """Set chat history."""
        self.chat_store.extend(messages)
        self._token_count += self._token_count_for_messages(messages)

    def reset(self) -> None:
        """Reset chat history."""
        self.chat_store.clear()

    def get_token_count(self) -> int:
        """Returns the token count of the memory buffer (excluding the last assistant response)."""
        return self._token_count

    def _token_count_for_messages(self, messages: List[Dict]) -> int:
        """Get token count for list of messages."""
        total = 0
        for m in messages:
            msg_str = m["content"]
            try:
                response = self.tokenizer_fn(msg_str)
                total += len(response)
            except Exception as e:
                print(e)
                total += 32768  # max tokenizer content-length

        return total

    def _split_messages_summary_or_full_text(self, chat_history: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """Determine which messages will be included as full text,
        and which will have to be summarized by the llm.
        """
        chat_history_full_text: List[Dict] = []
        
        full_messages_token_limit = self.token_limit // 3

        total_full_len = 0

        while True:
            if len(chat_history) <= 0:
                break            
            last_message_len = self._token_count_for_messages([chat_history[-1]])
            #if (self._token_count + last_message_len > self.token_limit):
            if (self._token_count + last_message_len > full_messages_token_limit):
                if len(chat_history_full_text) == 0 and last_message_len < full_messages_token_limit * 2:
                    # keep the last message even when bigger but not too big
                    pass
                else:
                    break

            total_full_len += last_message_len

            # traverse the history in reverse order, when token limit is about to be
            # exceeded, we stop, so remaining messages are summarized
            self._token_count += last_message_len
            chat_history_full_text.insert(0, chat_history.pop())        

        # get token-len for chat_history
        first_len = self._token_count_for_messages(chat_history)
        if first_len + total_full_len > self.token_limit:
            chat_history_to_be_summarized = chat_history.copy()
        else:
            chat_history_to_be_summarized = []            
            chat_history_full_text = chat_history + chat_history_full_text
        #self._handle_assistant_and_tool_messages(chat_history_full_text, chat_history_to_be_summarized)

        return chat_history_full_text, chat_history_to_be_summarized

    def _summarize_oldest_chat_history(self, chat_history_to_be_summarized: List[Dict]) -> Dict:
        """Use the llm to summarize the messages that do not fit into the
        buffer.
        """        
        # Only summarize if there is new information to be summarized
        if (len(chat_history_to_be_summarized) == 1 and chat_history_to_be_summarized[0]["role"] == "system"):
            return chat_history_to_be_summarized[0]
                
        sumarization = summarize_fn(chat_history_to_be_summarized)
        return {"role": "system", "content": sumarization}    


#********************* /Classes **************************

chat_memory = None
summary_agent = None

def init_chat_memory(token_limit: int):
    global chat_memory    
    global summary_agent
    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_MODEL_PATH, trust_remote_code=True)

    summary_agent = create_summary_agent()   
    
    #chat_memory = ChatSummaryMemoryBuffer.from_defaults(
    chat_memory = ChatSummaryMemoryBuffer(
        chat_history=[],        
        token_limit=token_limit,
        summarize_fn=summarize_fn,
        tokenizer_fn=tokenize_string,
        count_initial_tokens=True,
    )


last_history = []

def put_chat_message(msg: dict):    
    global chat_memory
    if "role" not in msg or "content" not in msg:
        return
    
    chat_memory.put(msg)

def get_chat_history():
    global chat_memory
    
    try:
        history = chat_memory.get()
        last_history = history  # save the history in case of later errors
        #pass
    except Exception as e:        
        print(f"Chat-Memory Error: {e.msg if hasattr(e, "msg") else str(e)}")
        return last_history

    return history

def get_chat_history_token_count():
    global chat_memory
    return chat_memory.get_token_count()