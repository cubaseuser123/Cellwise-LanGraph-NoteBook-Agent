"""
IPython Startup Script for Notebook Auto-Documentation.

This script is loaded automatically when IPython/Jupyter starts.
It registers a post_run_cell hook that triggers documentation generation.

Installation:
    Copy this file to: ~/.ipython/profile_default/startup/

The script only activates when running in a Jupyter notebook environment.
"""
import os
import sys
from pathlib import Path

# Add the project src to path (adjust path as needed)
# We need to find the specific project root where Notebook_Agent is located
# Since this runs in ~/.ipython/..., we need a fixed path or a way to find it.
# For this specific user environment, we know it is at d:\NextJs Projects\Notebook_Agent
PROJECT_ROOT = Path(r"d:\NextJs Projects\Notebook_Agent")
SRC_PATH = PROJECT_ROOT / "src"

if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


def _is_jupyter_environment() -> bool:
    """Check if we're running in a Jupyter notebook."""
    try:
        from IPython import get_ipython
        shell = get_ipython()
        if shell is None:
            return False
        
        # Check for Jupyter kernel
        shell_class = shell.__class__.__name__
        return shell_class in ("ZMQInteractiveShell", "TerminalInteractiveShell")
    except ImportError:
        return False


def _get_notebook_path() -> str:
    """
    Get the path to the current notebook.
    
    This is tricky because IPython doesn't directly expose the notebook path.
    We try multiple methods.
    """
    try:
        # Method 1: Check __session__ variable (set by some Jupyter setups)
        from IPython import get_ipython
        ip = get_ipython()
        
        if hasattr(ip, "user_ns") and "__session__" in ip.user_ns:
            return ip.user_ns["__session__"]
        
        # Method 2: Try to get from kernel connection file
        import json
        
        connection_file = ip.kernel.config.get("IPKernelApp", {}).get("connection_file", "")
        if connection_file:
            # The connection file is in the same directory as runtime info
            # We can try to find the notebook from Jupyter's API
            pass
        
        # Method 3: Use working directory with a generic name
        cwd = os.getcwd()
        # Look for .ipynb files in the current directory
        notebooks = list(Path(cwd).glob("*.ipynb"))
        if len(notebooks) == 1:
            return str(notebooks[0])
        
        # Fallback: Use a hash of the kernel id
        kernel_id = getattr(ip.kernel, "ident", "unknown")
        return str(Path(cwd) / f"notebook_{kernel_id}.ipynb")
        
    except Exception as e:
        # Ultimate fallback
        return str(Path(os.getcwd()) / "unknown_notebook.ipynb")


def _get_notebook_name(notebook_path: str) -> str:
    """Extract notebook name from path."""
    return Path(notebook_path).stem


def _extract_output(result) -> str:
    """Extract string representation of cell output."""
    try:
        if result.result is not None:
            output = str(result.result)
            # Truncate very long outputs
            if len(output) > 500:
                output = output[:500] + "... (truncated)"
            return output
    except Exception:
        pass
    return ""


def _setup_documentation_hook():
    """Set up the post_run_cell hook for documentation."""
    try:
        from IPython import get_ipython
        # We wrap imports in try/except because the project might not be fully built yet
        # or the path might be incorrect, and we don't want to crash Jupyter start
        try:
            from manager.notebook_manager import NotebookManager
            from agent.state import CellInfo
        except ImportError:
            # Silently fail or simple log if modules aren't ready
            return
        
    except ImportError as e:
        print(f"[NotebookDocs] Failed to import modules: {e}")
        print("[NotebookDocs] Documentation system disabled.")
        return
    
    ip = get_ipython()
    if ip is None:
        return
    
    # Get the manager instance
    try:
        manager = NotebookManager.get_instance()
    except Exception:
        return
    
    def post_run_cell_hook(result):
        """
        Called after each cell execution.
        
        Args:
            result: ExecutionResult object with cell info
        """
        # Skip failed cells
        if not result.success:
            return
        
        # Skip if no raw cell (shouldn't happen but be safe)
        if not hasattr(result.info, "raw_cell") or not result.info.raw_cell:
            return
        
        try:
            # Get notebook info
            notebook_path = _get_notebook_path()
            notebook_name = _get_notebook_name(notebook_path)
            
            # Build cell info
            cell_info: CellInfo = {
                "code": result.info.raw_cell,
                "output": _extract_output(result),
                "execution_count": ip.execution_count,
            }
            
            # Process through the agent (runs in background)
            manager.process_cell(notebook_path, notebook_name, cell_info)
            
        except Exception as e:
            # Never crash the notebook
            print(f"[NotebookDocs] Error in hook: {e}")
    
    # Register the hook
    ip.events.register("post_run_cell", post_run_cell_hook)
    print("[NotebookDocs] ✓ Auto-documentation enabled")


# Only set up if in Jupyter environment
if _is_jupyter_environment():
    try:
        _setup_documentation_hook()
    except Exception as e:
        print(f"[NotebookDocs] Setup failed: {e}")
