#checking here if we should process the cell for documentation or not. so stripping off anything unecessary like imports and such

import re 
from typing import Tuple, Optional

IMPORT_ONLY_PATTERN = re.compile(
    r"^\s*(import\s+\w+|from\s+\w+\s+import\s+.+)\s*$",
    re.MULTILINE
)
MAGIC_COMMAND_PATTERN = re.compile(r"^\s*[%!]")
PRINT_ONLY_PATTERN = re.compile(r"^\s*print\s*\(.*\)\s*$")
COMMENT_ONLY_PATTERN = re.compile(r"^\s*#.*$", re.MULTILINE)
DISPLAY_PATTERN = re.compile(r"^\s*(display\s*\(|plt\.show\s*\()")

def is_significant_cell(code:str) -> Tuple[bool, Optional[str]]:
    """
    Analyze if a cell warrants documentation.
    
    Args:
        code: The cell's source code
    
    Returns:
        Tuple of (is_significant, skip_reason)
        skip_reason is None if the cell is significant
    """

    #code is stripped for analysis
    stripped = code.strip()

    if not stripped:
        return False, 'empty_cell'

    if MAGIC_COMMAND_PATTERN.match(stripped):
        return False, "magic_command"

    #split into lines for line by line analysis
    lines = [l.strip() for l in stripped.split('\n') if l.strip()]

    if.all(l.startswith('#') for l in lines):
        return False, 'comment_only'

    #now the import only cells
    non_comment_lines = [l for l in lines if not l.startswith('#')]
    if all(IMPORT_ONLY_PATTERN.match(l) for l in non_comment_lines):
        return False, 'import_only'

    #now any single print statement
    if len(non_comment_lines) == 1 and PRINT_ONLY_PATTERN.match(non_comment_lines[0]):
        return False, 'print_only'

    #now just display/show calls
    if len(non_comment_lines) == 1 and DISPLAY_PATTERN.match(non_comment_lines[0]):
        return False, 'display_only'
    
    return True, None

def get_skip_reason_message(reason:str) -> str:
    """
    Returns a human-readable message for a skip reason.
    """
    messages = {
        'empty_cell': "Empty cell - no documentation needed.",
        'magic_command': "Cell contains only magic commands - no documentation needed.",
        'comment_only': "Cell contains only comments - no documentation needed.",
        'import_only': "Cell contains only imports - no documentation needed.",
        'print_only': "Cell contains only print statements - no documentation needed.",
        'display_only': "Cell contains only display/show calls - no documentation needed.",
    }
    return messages.get(reason, f"Skipped: {reason}")
