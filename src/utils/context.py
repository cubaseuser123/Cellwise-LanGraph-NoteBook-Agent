#Here we gather context of previous cells for our LLM to work with 
from typing import List
from agent.state import CellInfo
from config import get_settings

def format_context_for_prompt(cells: List[CellInfo]) -> str:
    """
    So we format cells into a context string so that our LLM can read it. 

    Args:
        cells: List of previous CellInfo objects
    
    Returns:
        Formatter string that will describe the context
    """
    if not cells:
        return ""

    settings = get_settings()
    max_length = settings.max_cell_length

    context_parts = []

    for cell in cells:
        code = cell['code']
        exec_count = cell['execution_count']

        if len(code) > max_length:
            code = code[:max_length] + "\n... (truncated)"

        context_parts.append(f"### Cell [{exec_count}]\n```python\n{code}\n```")
    return "\n\n".join(context_parts)

def summarize_cell(cell : CellInfo, max_lines: int = 5) -> str:
    """
    Here we create a brief summary of a cell for display.

    Args:
        cell: CellInfo to summarize
        max_lines: Maximum lines to show

    Returns:
        Summarized cell content
    """
    lines = cell['code'].split('\n')

    if len(lines) <= max_lines:
        return cell['code']

    shown = lines[:max_lines]
    remaining = len(lines) - max_lines

    return '\n'.join(shown) + f'\n... ({remaining} more lines)'