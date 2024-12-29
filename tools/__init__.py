# __init__.py
from .execute_python import execute_python_code, list_installed_python_libraries
from .web_search import search_news, search_the_web, open_and_read_webpage_from_url, read_pdf_from_url
from .memory import init_long_term_memory, store_in_memory_tool, recall_from_memory_tool
from .basic_tools import get_local_datetime, get_utc_datetime, clap_hands
from .file_tools import list_dirs, read_text_file, write_text_file, delete_file, make_dir, delete_dir
#from .text2speech import speak