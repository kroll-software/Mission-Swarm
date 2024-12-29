import os
from pathlib import Path
from directory_tree import DisplayTree

workspace = "./workspace"

def list_dirs() -> str:    
    ''' List directiries and files in the workspace '''
    print("Using Tool: list_dirs")
    str = DisplayTree(workspace, stringRep=True, showHidden=True, sortBy=2)
    str = "root\n" + str[len(workspace) : ]
    return str

def fix_path(file_path: str) -> str:
    if len(file_path) > 0 and file_path[0] == os.pathsep:
        file_path = file_path[1:]
    file_path = os.path.abspath(os.path.join(workspace, file_path))
    if not Path(file_path).is_relative_to(os.path.abspath(workspace)):
        raise ValueError("invalid path")
    return file_path

def read_text_file(file_path: str) -> str:
    ''' Read a text-file (*.txt, *.md, *.py, ...) from the workspace '''
    print("Using Tool: read_file")
    try:
        file_path = fix_path(file_path)
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)

def write_text_file(file_path: str, content: str) -> str:
    ''' Write a text-file (*.txt, *.md, *.py, ...) to the workspace '''
    print("Using Tool: write_file")
    try:             
        file_path = fix_path(file_path)
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return "OK"
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)  

def delete_file(file_path: str) -> str:
    ''' Delete a file from the workspace '''
    print("Using Tool: delete_file")
    try:
        file_path = fix_path(file_path)
        os.remove(file_path)
        return f"File {file_path} deleted successfully."
    except FileNotFoundError:
        return f"Error: File {file_path} not found."
    except PermissionError:
        return f"Error: Permission denied to delete file {file_path}."
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)

def make_dir(path: str) -> str:
    ''' Make a new directory in the workspace '''
    print("Using Tool: make_dir")
    try:                
        path = fix_path(path)
        os.mkdir(path)
        return "OK"
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)
    
def delete_dir(path: str) -> str:
    ''' Delete a directory in the workspace '''
    print("Using Tool: delete_dir")
    try:                
        path = fix_path(path)
        os.rmdir(path)
        return "OK"
    except Exception as e:
        return e.msg if hasattr(e, "msg") else str(e)