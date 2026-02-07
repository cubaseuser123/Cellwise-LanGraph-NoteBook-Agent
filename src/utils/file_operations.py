#all kind of file operations for markdown 

import os 
from pathlib import Path
from datetime import datetime
from filelock import FileLock
from typing import Optional

def get_docs_filepath(notebook_path:str)->str:
    """
    First we will convert notebook's file path to md's file path so that we can go from there

    Args:
        notebook_path: The path to the notebook file
    
    Returns:
        The path to the markdown file
    """
    path = Path(notebook_path)
    docs_name = f"{path.stem}_docs.md"
    return str(path.parent / docs_name)

def read_current_docs(filepath:str) -> str:
    """
    Read the current contents of that md file. 

    Args:
        filepath: The path to the markdown file
    
    Returns:
        The current contents of the markdown file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"[NotebookDocs] Error reading {filepath}: {e}")
        return ""

def append_to_docs(filepath: str, content: str) -> None:
    """
    Here we will append content to that md but safely, so file locking will be done. No concurrent writing allowed.

    Args:
        filepath: Path to the documentation file
        content: Content to append
    """
    lock_path = f"{filepath}.lock"
    lock = FileLock(lock_path, timeout=10)

    try:
        with lock:
            with open(filepath, "a", encoding='utf-8') as f:
                f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to write to {filepath}: {e}")

def initatlize_docs_file(filepath:str, notebook_name:str) -> None:
    """
    Here we create the md file if it does not exist yet

    Args:
    filepath: Path to the documentation file
    notebook_name: Name of the notebook
    """
    if os.path.exists(filepath):
        return
    
    header = f"# Documentation: {notebook_name}\n> Auto-generated documentation for Jupyter notebook cells.\n> Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"

    try:
        with open(filepath, "w", encoding='utf-8') as f:
            f.write(header)
    except Exception as e:
        print(f"[NotebookDocs] Error initializing {filepath}: {e}")
