"""
IPython Startup Script for Notebook Auto-Documentation.

This script is loaded automatically when IPython/Jupyter starts.
It registers a post_run_cell hook that triggers documentation generation.
"""

import os 
import sys 
from pathlib import Path

# PROJECT_ROOT = Path(__file__).parent.parent
# Hardcoded for local dev since script is copied to global profile
PROJECT_ROOT = Path(r"d:\NextJs Projects\Notebook_Agent")
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))

def _is_jupyter_environment() -> bool:
    #here we see if we are running in a jupyter environment
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        
        shell_class = shell.__class__.__name__
        return shell_class in ('ZMQInteractiveShell', 'TerminalInteractiveShell')
    except ImportError:
        return False

def _get_notebook_path() -> str:
    #here we get path to the current notebook.
    try:
        from IPython import get_ipython
        ip = get_ipython()

        cwd = os.getcwd()
        notebooks = list(Path(cwd).glob("*.ipynb"))
        
        # If there's only one notebook, it's definitely that one
        if len(notebooks) == 1:
            return str(notebooks[0])

        # If multiple, try to match using session or kernel ID
        if hasattr(ip, "user_ns") and "__session__" in ip.user_ns:
            return ip.user_ns["__session__"]
        
        connection_file = ip.kernel.config.get("IPKernelApp", {}).get("connection_file", "")
        if connection_file:
            # Could parse connection file to get kernel ID
            pass

        kernel_id = getattr(ip.kernel, "ident", "unkown")
        return str(Path(cwd) / f"notebook_{kernel_id}.ipynb")
    except Exception as e:
        return str(Path(os.getcwd()) / "unknown_notebook.ipynb")

def _get_notebook_name(notebook_path: str) -> str:
    return Path(notebook_path).stem

def _extract_output(result) -> str:
    try:
        if result.result is not None:
            output = str(result.result)
            if len(output) > 500:
                output = output[:500] + "... (truncated)"
            return output
    except Exception:
        pass
    return ""

def _setup_documentation_hook():
    try:
        from IPython import get_ipython
        from manager.notebook_manager import NotebookManager
        from agent.state import CellInfo
    
    except ImportError as e:
        print(f"[NotebookDocs] Failed to import modules: {e}")
        print("[NotebookDocs] Documentation system disabled.")
        return 

    ip = get_ipython()
    if ip is None:
        return 

    manager = NotebookManager.get_instance()

    def post_run_cell_hook(result):
        if not result.success:
            return 
        
        if not hasattr(result.info, "raw_cell") or not result.info.raw_cell:
            return
        
        try:
            notebook_path = _get_notebook_path()
            notebook_name = _get_notebook_name(notebook_path)

            cell_info = {
                "code" : result.info.raw_cell,
                "output" : _extract_output(result),
                "execution_count" : ip.execution_count,
            }

            manager.process_cell(notebook_path, notebook_name, cell_info)

        except Exception as e:
            print(f"[NotebookDocs] Error in hook: {e}")
        
    ip.events.register("post_run_cell", post_run_cell_hook)
    print("[NotebookDocs] Auto-documentation enabled")

if _is_jupyter_environment():
    try:
        _setup_documentation_hook()
    except Exception as e:
        print(f"[NotebookDocs] setup failed : {e}")

print("[NotebookDocs] Startup script loaded successfully!")
