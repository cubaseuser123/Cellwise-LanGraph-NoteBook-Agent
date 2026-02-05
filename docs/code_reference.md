# Notebook Agent - Code Reference

Complete code reference for implementing the Jupyter Notebook Auto-Documentation System.

---

## Table of Contents

1. [Project Setup](#project-setup)
2. [Folder Structure](#folder-structure)
3. [Dependencies](#dependencies)
4. [Configuration Files](#configuration-files)
5. [Agent State](#agent-state)
6. [Agent Nodes](#agent-nodes)
7. [Graph Definition](#graph-definition)
8. [Notebook Manager](#notebook-manager)
9. [Utility Functions](#utility-functions)
10. [IPython Startup Script](#ipython-startup-script)
11. [Installation & Setup](#installation--setup)

---

## Project Setup

### 1. Create Project Structure

```bash
# Create main directories
mkdir -p src/agent src/manager src/utils src/config startup tests docs

# Create __init__.py files
touch src/__init__.py
touch src/agent/__init__.py
touch src/manager/__init__.py
touch src/utils/__init__.py
touch src/config/__init__.py
touch tests/__init__.py
```

---

## Folder Structure

```
Notebook_Agent/
├── docs/
│   ├── implementation_plan.md    # High-level plan
│   └── code_reference.md         # This file
├── src/
│   ├── __init__.py
│   ├── agent/
│   │   ├── __init__.py
│   │   ├── state.py              # AgentState TypedDict
│   │   ├── nodes.py              # 5 node functions
│   │   └── graph.py              # LangGraph workflow
│   ├── manager/
│   │   ├── __init__.py
│   │   └── notebook_manager.py   # Singleton manager
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── significance.py       # Cell filter logic
│   │   ├── file_ops.py           # Markdown I/O
│   │   └── context.py            # Context formatting
│   └── config/
│       ├── __init__.py
│       └── settings.py           # Pydantic settings
├── startup/
│   └── 00_notebook_docs.py       # IPython hook
├── tests/
│   ├── __init__.py
│   ├── test_significance.py
│   ├── test_nodes.py
│   ├── test_agent.py
│   └── test_manager.py
├── .env.example
├── .env                          # Your actual credentials (gitignored)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Dependencies

### requirements.txt

```txt
# LangGraph & LangChain
langgraph>=0.2.0
langchain-core>=0.3.0

# OpenAI SDK (for Vercel AI Gateway)
openai>=1.0.0

# Configuration
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# File locking for concurrent writes
filelock>=3.0.0

# IPython (usually pre-installed with Jupyter)
ipython>=8.0.0

# Testing
pytest>=7.0.0
pytest-asyncio>=0.21.0
```

### Installation Command

```bash
pip install -r requirements.txt
```

---

## Configuration Files

### .env.example

```env
# Vercel AI Gateway Configuration
VERCEL_AI_GATEWAY_URL=https://gateway.ai.vercel.sh/v1
VERCEL_AI_GATEWAY_API_KEY=your_api_key_here

# Model Configuration
MODEL_NAME=mistral/devstral-2
MODEL_TEMPERATURE=0.7
MODEL_MAX_TOKENS=1000

# Context Settings
MAX_CONTEXT_CELLS=5
MAX_CELL_LENGTH=2000
```

### src/config/settings.py

```python
"""
Configuration management using pydantic-settings.
Loads values from .env file and environment variables.
"""
from pydantic_settings import BaseSettings
from pydantic import Field
from functools import lru_cache


class Settings(BaseSettings):
    """Application settings loaded from environment."""
    
    # Vercel AI Gateway
    vercel_ai_gateway_url: str = Field(
        default="https://gateway.ai.vercel.sh/v1",
        description="Vercel AI Gateway endpoint URL"
    )
    vercel_ai_gateway_api_key: str = Field(
        ...,  # Required field
        description="API key for Vercel AI Gateway"
    )
    
    # Model configuration
    model_name: str = Field(
        default="mistral/devstral-2",
        description="Model identifier in provider/model format"
    )
    model_temperature: float = Field(
        default=0.7,
        ge=0.0,
        le=2.0,
        description="Sampling temperature"
    )
    model_max_tokens: int = Field(
        default=1000,
        ge=100,
        le=4000,
        description="Maximum tokens in response"
    )
    
    # Context settings
    max_context_cells: int = Field(
        default=5,
        ge=1,
        le=10,
        description="Number of previous cells for context"
    )
    max_cell_length: int = Field(
        default=2000,
        description="Max characters per cell before truncation"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
```

### src/config/__init__.py

```python
from .settings import Settings, get_settings

__all__ = ["Settings", "get_settings"]
```

---

## Agent State

### src/agent/state.py

```python
"""
LangGraph agent state definition.
Defines the structure of data flowing through the graph.
"""
from typing import TypedDict, Optional, List


class CellInfo(TypedDict):
    """Information about a single notebook cell."""
    code: str
    output: Optional[str]
    execution_count: int


class AgentState(TypedDict):
    """
    State object passed through the LangGraph workflow.
    All nodes read from and write to this state.
    """
    # Current cell being processed
    current_cell: CellInfo
    notebook_name: str
    notebook_path: str
    
    # Context from previous cells
    previous_cells: List[CellInfo]
    
    # Processing flags (set by check_significance)
    is_significant: bool
    skip_reason: Optional[str]
    
    # Generated content
    explanation: Optional[str]
    formatted_markdown: Optional[str]
    
    # File tracking
    docs_filepath: str
    current_docs_content: Optional[str]
```

### src/agent/__init__.py

```python
from .state import AgentState, CellInfo
from .nodes import (
    check_significance,
    gather_context,
    generate_explanation,
    format_markdown,
    update_file,
)
from .graph import create_documentation_graph

__all__ = [
    "AgentState",
    "CellInfo",
    "check_significance",
    "gather_context",
    "generate_explanation",
    "format_markdown",
    "update_file",
    "create_documentation_graph",
]
```

---

## Agent Nodes

### src/agent/nodes.py

````python
"""
LangGraph node implementations.
Each function is a node in the documentation workflow.
"""
import asyncio
from openai import AsyncOpenAI
from typing import Dict, Any

from ..config import get_settings
from ..utils.significance import is_significant_cell
from ..utils.file_ops import append_to_docs, read_current_docs
from ..utils.context import format_context_for_prompt
from .state import AgentState


_client: AsyncOpenAI = None


def _get_client() -> AsyncOpenAI:
    """Lazy initialization of OpenAI client for Vercel AI Gateway."""
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            base_url=settings.vercel_ai_gateway_url,
            api_key=settings.vercel_ai_gateway_api_key,
        )
    return _client


def check_significance(state: AgentState) -> Dict[str, Any]:
    """Filter out trivial cells that don't warrant documentation."""
    code = state["current_cell"]["code"]
    is_significant, skip_reason = is_significant_cell(code)
    return {"is_significant": is_significant, "skip_reason": skip_reason}


def gather_context(state: AgentState) -> Dict[str, Any]:
    """Read current docs file content for context."""
    docs_filepath = state["docs_filepath"]
    current_content = read_current_docs(docs_filepath)
    return {"current_docs_content": current_content}


async def generate_explanation(state: AgentState) -> Dict[str, Any]:
    """Call Vercel AI Gateway (Mistral Devstral 2) to explain the code."""
    settings = get_settings()
    client = _get_client()
    
    current_code = state["current_cell"]["code"]
    context = format_context_for_prompt(state["previous_cells"])
    
    if len(current_code) > settings.max_cell_length:
        current_code = current_code[:settings.max_cell_length] + "\n... (truncated)"
    
    system_prompt = """You are a code documentation expert. Explain what the given Python code does in clear, concise terms (2-4 sentences). Focus on what it accomplishes, key operations, and how it relates to previous context."""

    user_prompt = f"""## Previous Context
{context if context else "(No previous context)"}

## Current Cell (Execution #{state['current_cell']['execution_count']})
```python
{current_code}
```

Explain what this code does:"""

    try:
        response = await client.chat.completions.create(
            model=settings.model_name,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=settings.model_temperature,
            max_tokens=settings.model_max_tokens,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"*Documentation generation failed: {str(e)}*"
    
    return {"explanation": explanation}


def format_markdown(state: AgentState) -> Dict[str, Any]:
    """Structure the explanation into proper markdown format."""
    cell = state["current_cell"]
    markdown = f"""
---

## Cell [{cell['execution_count']}]

```python
{cell['code']}
```

### Explanation

{state['explanation']}

"""
    return {"formatted_markdown": markdown}


def update_file(state: AgentState) -> Dict[str, Any]:
    """Append the formatted markdown to the documentation file."""
    try:
        append_to_docs(state["docs_filepath"], state["formatted_markdown"])
    except Exception as e:
        print(f"[NotebookDocs] Error writing docs: {e}")
    return {}


# Sync wrapper: LangGraph runs nodes synchronously, but generate_explanation is async.
# This wrapper bridges the gap using asyncio.run().
def generate_explanation_sync(state: AgentState) -> Dict[str, Any]:
    """Synchronous wrapper for generate_explanation (required by LangGraph)."""
    return asyncio.run(generate_explanation(state))
````

---

## Graph Definition

> ⚠️ **This is `src/agent/graph.py`** — the LangGraph workflow that connects all the nodes above.

### src/agent/graph.py

```python
"""
LangGraph workflow definition.
Compiles the documentation agent graph.
"""
from langgraph.graph import StateGraph, END
from typing import Literal

from .state import AgentState
from .nodes import (
    check_significance,
    gather_context,
    generate_explanation_sync,
    format_markdown,
    update_file,
)


def should_continue(state: AgentState) -> Literal["gather_context", "end"]:
    """
    Conditional edge after check_significance.
    Routes to next node or ends early for insignificant cells.
    """
    if state["is_significant"]:
        return "gather_context"
    else:
        return "end"


def create_documentation_graph() -> StateGraph:
    """
    Create and compile the documentation workflow graph.
    
    Flow:
        START -> check_significance -> (conditional)
                                      |-> gather_context -> generate_explanation 
                                      |                     -> format_markdown -> update_file -> END
                                      |-> END (if not significant)
    
    Returns compiled StateGraph.
    """
    # Create the graph with our state schema
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("check_significance", check_significance)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("generate_explanation", generate_explanation_sync)
    workflow.add_node("format_markdown", format_markdown)
    workflow.add_node("update_file", update_file)
    
    # Set entry point
    workflow.set_entry_point("check_significance")
    
    # Add conditional edge after significance check
    workflow.add_conditional_edges(
        "check_significance",
        should_continue,
        {
            "gather_context": "gather_context",
            "end": END,
        }
    )
    
    # Add sequential edges for the main flow
    workflow.add_edge("gather_context", "generate_explanation")
    workflow.add_edge("generate_explanation", "format_markdown")
    workflow.add_edge("format_markdown", "update_file")
    workflow.add_edge("update_file", END)
    
    # Compile and return
    return workflow.compile()


# Pre-compiled graph for reuse
_compiled_graph = None


def get_documentation_graph():
    """Get or create the compiled documentation graph."""
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_documentation_graph()
    return _compiled_graph
```

---

## Notebook Manager

### src/manager/notebook_manager.py

```python
"""
Notebook Manager - Singleton for tracking multiple open notebooks.
Manages agent instances and cell history per notebook.
"""
from typing import Dict, List, Optional
from pathlib import Path
import threading

from ..agent.state import CellInfo
from ..agent.graph import get_documentation_graph
from ..utils.file_ops import get_docs_filepath, initialize_docs_file
from ..config import get_settings


class NotebookManager:
    """
    Singleton manager for tracking notebooks and their agents.
    Thread-safe for concurrent cell executions.
    """
    
    _instance: Optional["NotebookManager"] = None
    _lock = threading.Lock()
    
    def __new__(cls) -> "NotebookManager":
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        # Cell history per notebook path
        self._cell_history: Dict[str, List[CellInfo]] = {}
        
        # Lock for thread-safe operations
        self._history_lock = threading.Lock()
        
        self._initialized = True
    
    @classmethod
    def get_instance(cls) -> "NotebookManager":
        """Get the singleton instance."""
        return cls()
    
    def record_cell(self, notebook_path: str, cell: CellInfo) -> None:
        """
        Record a cell execution for context tracking.
        
        Args:
            notebook_path: Absolute path to the notebook
            cell: Cell information to record
        """
        with self._history_lock:
            if notebook_path not in self._cell_history:
                self._cell_history[notebook_path] = []
            
            self._cell_history[notebook_path].append(cell)
            
            # Keep only recent cells to limit memory
            settings = get_settings()
            max_cells = settings.max_context_cells * 2  # Keep some buffer
            if len(self._cell_history[notebook_path]) > max_cells:
                self._cell_history[notebook_path] = \
                    self._cell_history[notebook_path][-max_cells:]
    
    def get_recent_cells(
        self, 
        notebook_path: str, 
        n: Optional[int] = None
    ) -> List[CellInfo]:
        """
        Get the n most recent cells for a notebook.
        
        Args:
            notebook_path: Absolute path to the notebook
            n: Number of cells to return (default from settings)
        
        Returns:
            List of recent CellInfo objects
        """
        if n is None:
            n = get_settings().max_context_cells
        
        with self._history_lock:
            history = self._cell_history.get(notebook_path, [])
            # Return all but the last one (which is the current cell)
            return history[-(n+1):-1] if len(history) > 1 else []
    
    def process_cell(
        self,
        notebook_path: str,
        notebook_name: str,
        cell: CellInfo,
    ) -> None:
        """
        Process a cell execution through the documentation agent.
        
        Args:
            notebook_path: Absolute path to the notebook
            notebook_name: Name of the notebook (without extension)
            cell: Cell information to process
        """
        # Record the cell first
        self.record_cell(notebook_path, cell)
        
        # Get docs file path
        docs_filepath = get_docs_filepath(notebook_path)
        
        # Initialize docs file if it doesn't exist
        initialize_docs_file(docs_filepath, notebook_name)
        
        # Get previous cells for context
        previous_cells = self.get_recent_cells(notebook_path)
        
        # Build initial state
        initial_state = {
            "current_cell": cell,
            "notebook_name": notebook_name,
            "notebook_path": notebook_path,
            "previous_cells": previous_cells,
            "is_significant": False,
            "skip_reason": None,
            "explanation": None,
            "formatted_markdown": None,
            "docs_filepath": docs_filepath,
            "current_docs_content": None,
        }
        
        # Run the graph
        graph = get_documentation_graph()
        try:
            graph.invoke(initial_state)
        except Exception as e:
            # Don't crash the notebook on documentation errors
            print(f"[NotebookDocs] Error processing cell: {e}")
    
    def cleanup_notebook(self, notebook_path: str) -> None:
        """
        Clean up resources for a closed notebook.
        
        Args:
            notebook_path: Absolute path to the notebook
        """
        with self._history_lock:
            if notebook_path in self._cell_history:
                del self._cell_history[notebook_path]
    
    def reset(self) -> None:
        """Reset all state (useful for testing)."""
        with self._history_lock:
            self._cell_history.clear()
```

### src/manager/__init__.py

```python
from .notebook_manager import NotebookManager

__all__ = ["NotebookManager"]
```

---

## Utility Functions

### src/utils/significance.py

```python
"""
Cell significance checker.
Determines if a cell warrants documentation.
"""
import re
from typing import Tuple, Optional


# Patterns for insignificant cells
IMPORT_ONLY_PATTERN = re.compile(
    r"^\s*(import\s+\w+|from\s+\w+\s+import\s+.+)\s*$",
    re.MULTILINE
)
MAGIC_COMMAND_PATTERN = re.compile(r"^\s*[%!]")
PRINT_ONLY_PATTERN = re.compile(r"^\s*print\s*\(.*\)\s*$")
COMMENT_ONLY_PATTERN = re.compile(r"^\s*#.*$", re.MULTILINE)
DISPLAY_PATTERN = re.compile(r"^\s*(display\s*\(|plt\.show\s*\()")


def is_significant_cell(code: str) -> Tuple[bool, Optional[str]]:
    """
    Analyze if a cell warrants documentation.
    
    Args:
        code: The cell's source code
    
    Returns:
        Tuple of (is_significant, skip_reason)
        skip_reason is None if the cell is significant
    """
    # Strip the code for analysis
    stripped = code.strip()
    
    # Check 1: Empty or whitespace only
    if not stripped:
        return False, "empty_cell"
    
    # Check 2: Magic commands (%, !, %%)
    if MAGIC_COMMAND_PATTERN.match(stripped):
        return False, "magic_command"
    
    # Split into lines for line-by-line analysis
    lines = [l.strip() for l in stripped.split("\n") if l.strip()]
    
    # Check 3: Comment-only cells
    if all(l.startswith("#") for l in lines):
        return False, "comment_only"
    
    # Check 4: Import-only cells
    non_comment_lines = [l for l in lines if not l.startswith("#")]
    if all(IMPORT_ONLY_PATTERN.match(l) for l in non_comment_lines):
        return False, "import_only"
    
    # Check 5: Single print statement
    if len(non_comment_lines) == 1 and PRINT_ONLY_PATTERN.match(non_comment_lines[0]):
        return False, "print_only"
    
    # Check 6: Just display/show calls
    if len(non_comment_lines) == 1 and DISPLAY_PATTERN.match(non_comment_lines[0]):
        return False, "display_only"
    
    # If we get here, the cell is significant
    return True, None


def get_skip_reason_message(reason: str) -> str:
    """Get a human-readable message for a skip reason."""
    messages = {
        "empty_cell": "Cell is empty",
        "magic_command": "Cell contains only magic commands",
        "comment_only": "Cell contains only comments",
        "import_only": "Cell contains only import statements",
        "print_only": "Cell contains only a print statement",
        "display_only": "Cell contains only display/show calls",
    }
    return messages.get(reason, f"Skipped: {reason}")
```

### src/utils/file_ops.py

```python
"""
File operations for markdown documentation.
Handles reading, writing, and initializing doc files.
"""
import os
from pathlib import Path
from datetime import datetime
from filelock import FileLock
from typing import Optional


def get_docs_filepath(notebook_path: str) -> str:
    """
    Convert notebook path to documentation file path.
    
    Example: /path/to/notebook.ipynb -> /path/to/notebook_docs.md
    
    Args:
        notebook_path: Absolute path to the notebook
    
    Returns:
        Absolute path to the documentation file
    """
    path = Path(notebook_path)
    docs_name = f"{path.stem}_docs.md"
    return str(path.parent / docs_name)


def read_current_docs(filepath: str) -> str:
    """
    Read the current contents of a documentation file.
    
    Args:
        filepath: Path to the documentation file
    
    Returns:
        File contents or empty string if file doesn't exist
    """
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return ""
    except Exception as e:
        print(f"[NotebookDocs] Error reading {filepath}: {e}")
        return ""


def append_to_docs(filepath: str, content: str) -> None:
    """
    Safely append content to a documentation file.
    Uses file locking for concurrent access.
    
    Args:
        filepath: Path to the documentation file
        content: Content to append
    """
    lock_path = f"{filepath}.lock"
    lock = FileLock(lock_path, timeout=10)
    
    try:
        with lock:
            with open(filepath, "a", encoding="utf-8") as f:
                f.write(content)
    except Exception as e:
        raise RuntimeError(f"Failed to write to {filepath}: {e}")


def initialize_docs_file(filepath: str, notebook_name: str) -> None:
    """
    Create documentation file with header if it doesn't exist.
    
    Args:
        filepath: Path to the documentation file
        notebook_name: Name of the notebook (for the header)
    """
    if os.path.exists(filepath):
        return
    
    header = f"""# Documentation: {notebook_name}

> Auto-generated documentation for Jupyter notebook cells.
> Generated on: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

"""
    
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(header)
    except Exception as e:
        print(f"[NotebookDocs] Error initializing {filepath}: {e}")
```

### src/utils/context.py

```python
"""
Context formatting utilities.
Prepares previous cell information for LLM prompts.
"""
from typing import List
from ..agent.state import CellInfo
from ..config import get_settings


def format_context_for_prompt(cells: List[CellInfo]) -> str:
    """
    Format previous cells into a context string for the LLM.
    
    Args:
        cells: List of previous CellInfo objects
    
    Returns:
        Formatted string describing the context
    """
    if not cells:
        return ""
    
    settings = get_settings()
    max_length = settings.max_cell_length
    
    context_parts = []
    
    for cell in cells:
        code = cell["code"]
        exec_count = cell["execution_count"]
        
        # Truncate if needed
        if len(code) > max_length:
            code = code[:max_length] + "\n... (truncated)"
        
        context_parts.append(f"### Cell [{exec_count}]\n```python\n{code}\n```")
    
    return "\n\n".join(context_parts)


def summarize_cell(cell: CellInfo, max_lines: int = 5) -> str:
    """
    Create a brief summary of a cell for display.
    
    Args:
        cell: CellInfo to summarize
        max_lines: Maximum lines to show
    
    Returns:
        Summarized cell content
    """
    lines = cell["code"].split("\n")
    
    if len(lines) <= max_lines:
        return cell["code"]
    
    shown = lines[:max_lines]
    remaining = len(lines) - max_lines
    
    return "\n".join(shown) + f"\n... ({remaining} more lines)"
```

### src/utils/__init__.py

```python
from .significance import is_significant_cell, get_skip_reason_message
from .file_ops import get_docs_filepath, read_current_docs, append_to_docs, initialize_docs_file
from .context import format_context_for_prompt, summarize_cell

__all__ = [
    "is_significant_cell",
    "get_skip_reason_message",
    "get_docs_filepath",
    "read_current_docs",
    "append_to_docs",
    "initialize_docs_file",
    "format_context_for_prompt",
    "summarize_cell",
]
```

---

## IPython Startup Script

### startup/00_notebook_docs.py

```python
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
PROJECT_ROOT = Path(__file__).parent.parent
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
        from manager.notebook_manager import NotebookManager
        from agent.state import CellInfo
        
    except ImportError as e:
        print(f"[NotebookDocs] Failed to import modules: {e}")
        print("[NotebookDocs] Documentation system disabled.")
        return
    
    ip = get_ipython()
    if ip is None:
        return
    
    # Get the manager instance
    manager = NotebookManager.get_instance()
    
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
```

---

## Installation & Setup

### Step 1: Clone/Create Project

```bash
cd "d:\NextJs Projects\Notebook_Agent"
```

### Step 2: Create Virtual Environment

#### Option A: Using uv (Recommended - Faster)

[uv](https://docs.astral.sh/uv/) is a fast Python package installer and resolver.

```bash
# Install uv (if not already installed)
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Or using pip
pip install uv

# Create virtual environment with uv
uv venv

# Activate the virtual environment
.\.venv\Scripts\activate  # Windows
# or
source .venv/bin/activate  # Linux/Mac
```

#### Option B: Using Traditional venv

```bash
python -m venv venv
.\venv\Scripts\activate  # Windows
# or
source venv/bin/activate  # Linux/Mac
```

### Step 3: Install Dependencies

#### Using uv (Faster)

```bash
# Install from requirements.txt
uv pip install -r requirements.txt

# Or install individual packages
uv pip install langgraph langchain-core openai pydantic pydantic-settings python-dotenv filelock ipython pytest pytest-asyncio
```

#### Using pip (Traditional)

```bash
pip install -r requirements.txt
```

### Step 4: Configure Environment

```bash
# Copy example env file
copy .env.example .env

# Edit .env with your credentials
notepad .env
```

### Step 5: Install IPython Startup Script

```bash
# Find your IPython profile directory
python -c "import IPython; print(IPython.paths.get_ipython_dir())"

# Copy startup script (adjust paths as needed)
copy startup\00_notebook_docs.py %USERPROFILE%\.ipython\profile_default\startup\
```

### Step 6: Test the Installation

```bash
# Start Jupyter
jupyter notebook

# Create a new notebook and run a cell like:
# def hello(name):
#     return f"Hello, {name}!"
#
# Check that [notebook_name]_docs.md is created
```

---

## Testing

### Run All Tests

```bash
pytest tests/ -v
```

### Run Specific Test File

```bash
pytest tests/test_significance.py -v
pytest tests/test_nodes.py -v
```

### Test with Coverage

```bash
pytest tests/ --cov=src --cov-report=html
```
