# This file is to check for multiple open notebooks at once
# Multiple open notebooks = multiple agents

from typing import Dict, List, Optional
from pathlib import Path
import threading

from agent.state import CellInfo
from agent.graph import get_documentation_graph
from utils.file_operations import get_docs_filepath, initatlize_docs_file
from config import get_settings


class NotebookManager:
    _instance: Optional['NotebookManager'] = None
    _lock = threading.Lock()

    def __new__(cls) -> 'NotebookManager':
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        self._cell_history: Dict[str, List[CellInfo]] = {}
        self._history_lock = threading.Lock()
        self._initialized = True

    @classmethod
    def get_instance(cls) -> "NotebookManager":
        return cls()

    def record_cell(self, notebook_path: str, cell: CellInfo) -> None:
        """
        Here we record a cell execution for context tracking

        Args:
            notebook_path: Absolute path to the notebook
            cell: Cell information to record
        """
        with self._history_lock:
            if notebook_path not in self._cell_history:
                self._cell_history[notebook_path] = []
            self._cell_history[notebook_path].append(cell)

            settings = get_settings()
            max_cells = settings.max_context_cells * 2
            if len(self._cell_history[notebook_path]) > max_cells:
                self._cell_history[notebook_path] = \
                    self._cell_history[notebook_path][-max_cells:]

    def get_recent_cells(
        self,
        notebook_path: str,
        n: Optional[int] = None
    ) -> List[CellInfo]:
        """
        Get the n most recent cells for a notebook
        """
        if n is None:
            n = get_settings().max_context_cells

        with self._history_lock:
            history = self._cell_history.get(notebook_path, [])
            return history[-(n+1):-1] if len(history) > 1 else []

    def process_cell(
        self,
        notebook_path: str,
        notebook_name: str,
        cell: CellInfo,
    ) -> None:
        """
        Process a cell execution through the documentation pipeline
        """
        self.record_cell(notebook_path, cell)

        docs_filepath = get_docs_filepath(notebook_path)

        initatlize_docs_file(docs_filepath, notebook_name)

        previous_cells = self.get_recent_cells(notebook_path)

        initial_state = {
            "current_cell": cell,
            "notebook_name": notebook_name,
            "notebook_path": notebook_path,
            "previous_cells": previous_cells,
            "docs_filepath": docs_filepath,
            "current_docs_content": None,
            "is_significant": False,
            "skip_reason": None,
            "explanation": None,
            "formatted_markdown": None,
        }

        graph = get_documentation_graph()
        try:
            graph.invoke(initial_state)
        except Exception as e:
            print(f"[NotebookDocs] Error processing cell: {e}")

    def cleanup_notebook(self, notebook_path: str) -> None:
        """Clean up history for a notebook"""
        with self._history_lock:
            if notebook_path in self._cell_history:
                del self._cell_history[notebook_path]

    def reset(self) -> None:
        """Reset all notebook histories"""
        with self._history_lock:
            self._cell_history.clear()