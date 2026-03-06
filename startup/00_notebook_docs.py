"""
IPython Startup Script for Notebook Auto-Documentation.

This script is loaded automatically when IPython/Jupyter starts.
It registers a post_run_cell hook that triggers documentation generation.
"""

import os
import sys
from pathlib import Path

# Since this script is copied to the global IPython startup folder,
# __file__ no longer points to the project. We resolve the project root
# via the NOTEBOOK_AGENT_PATH environment variable.
#
# Set it in your shell profile OR in the .env file inside the project:
#   NOTEBOOK_AGENT_PATH=/path/to/Notebook_Agent

def _load_env_file():
    """Try to load a .env file from NOTEBOOK_AGENT_PATH if already in env,
    or search common locations as a bootstrap convenience."""
    try:
        from dotenv import load_dotenv

        # If NOTEBOOK_AGENT_PATH is already set, load its .env
        existing = os.environ.get("NOTEBOOK_AGENT_PATH")
        if existing:
            load_dotenv(Path(existing) / ".env", override=False)
            return

        # Bootstrap: search common locations for a .env that sets the var
        search_dirs = [Path.home(), Path.cwd(), Path.cwd().parent]
        for d in search_dirs:
            env_file = d / ".env"
            if env_file.exists():
                load_dotenv(env_file, override=False)
                if os.environ.get("NOTEBOOK_AGENT_PATH"):
                    return
    except ImportError:
        pass  # python-dotenv not available; rely on shell env

_load_env_file()

_project_root_str = os.environ.get("NOTEBOOK_AGENT_PATH")
if not _project_root_str:
    print(
        "[NotebookDocs] ERROR: NOTEBOOK_AGENT_PATH environment variable is not set.\n"
        "[NotebookDocs] Add it to your .env file or shell profile:\n"
        "[NotebookDocs]   NOTEBOOK_AGENT_PATH=/path/to/Notebook_Agent\n"
        "[NotebookDocs] Documentation system disabled."
    )
    # Define a sentinel so the rest of the script can bail out cleanly
    _SETUP_DISABLED = True
else:
    PROJECT_ROOT = Path(_project_root_str)
    SRC_PATH = PROJECT_ROOT / "src"
    if str(SRC_PATH) not in sys.path:
        sys.path.insert(0, str(SRC_PATH))
    _SETUP_DISABLED = False

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

def _find_notebook_by_code(current_code: str) -> str:
    """Find the notebook file that contains the given code."""
    import json
    cwd = os.getcwd()
    matches = []
    
    for nb_file in Path(cwd).glob("*.ipynb"):
        try:
            with open(nb_file, "r", encoding="utf-8") as f:
                content = json.load(f)
            
            for cell in content.get("cells", []):
                if cell.get("cell_type") == "code":
                    source = "".join(cell.get("source", []))
                    if source.strip() == current_code.strip():
                        matches.append(nb_file)
                        break
        except:
            pass
    
    if len(matches) == 1:
        return str(matches[0])
    
    if len(matches) > 1:
        # Prefer non-Untitled notebooks
        for m in matches:
            if "untitled" not in m.stem.lower():
                return str(m)
        return str(matches[0])
    
    return None

def _get_notebook_path(current_code: str = None) -> str:
    """Get path to the current notebook, using content matching if available."""
    try:
        from IPython import get_ipython
        ip = get_ipython()
        cwd = os.getcwd()
        
        # Strategy 1: Content matching (most reliable)
        if current_code:
            matched = _find_notebook_by_code(current_code)
            if matched:
                return matched
        
        # Strategy 2: Single notebook in directory
        notebooks = list(Path(cwd).glob("*.ipynb"))
        if len(notebooks) == 1:
            return str(notebooks[0])
        
        # Strategy 3: Session variable (fallback)
        if hasattr(ip, "user_ns") and "__session__" in ip.user_ns:
            return ip.user_ns["__session__"]
        
        # Strategy 4: Kernel ID (last resort)
        kernel_id = getattr(ip.kernel, "ident", "unknown")
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

    # Track the last known notebook path to detect renames
    _hook_state = {"last_notebook_path": None}

    def post_run_cell_hook(result):
        if not result.success:
            return

        if not hasattr(result.info, "raw_cell") or not result.info.raw_cell:
            return

        try:
            from utils.file_operations import get_docs_filepath

            current_code = result.info.raw_cell
            notebook_path = _get_notebook_path(current_code)
            notebook_name = _get_notebook_name(notebook_path)

            last_path = _hook_state["last_notebook_path"]

            # Detect a rename: path changed and the old one looked like a temp name
            if (
                last_path
                and last_path != notebook_path
                and Path(last_path).exists() is False  # old .ipynb is gone
            ):
                old_docs = get_docs_filepath(last_path)
                new_docs = get_docs_filepath(notebook_path)

                if Path(old_docs).exists() and not Path(new_docs).exists():
                    try:
                        Path(old_docs).rename(new_docs)
                        # Also migrate any lock file if present
                        old_lock = Path(old_docs + ".lock")
                        if old_lock.exists():
                            old_lock.unlink()
                        # Transfer cell history in manager
                        with manager._history_lock:
                            if last_path in manager._cell_history:
                                manager._cell_history[notebook_path] = \
                                    manager._cell_history.pop(last_path)
                        print(
                            f"[NotebookDocs] Notebook renamed — docs migrated to: "
                            f"{Path(new_docs).name}"
                        )
                    except Exception as rename_err:
                        print(f"[NotebookDocs] Could not migrate docs file: {rename_err}")

            _hook_state["last_notebook_path"] = notebook_path

            cell_info = {
                "code": result.info.raw_cell,
                "output": _extract_output(result),
                "execution_count": ip.execution_count,
            }

            # Run in a background thread so the kernel is never blocked
            import threading
            thread = threading.Thread(
                target=manager.process_cell,
                args=(notebook_path, notebook_name, cell_info),
                daemon=True,
            )
            thread.start()

        except Exception as e:
            print(f"[NotebookDocs] Error in hook: {e}")

    ip.events.register("post_run_cell", post_run_cell_hook)
    print("[NotebookDocs] Auto-documentation enabled")

if not _SETUP_DISABLED and _is_jupyter_environment():
    try:
        _setup_documentation_hook()
    except Exception as e:
        print(f"[NotebookDocs] setup failed : {e}")

print("[NotebookDocs] Startup script loaded successfully!")
