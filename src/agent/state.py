from typing import TypedDict, Optional, List

class CellInfo(TypedDict):
    code:str,
    output:Optional[str],
    execution_count:int

class AgentState(TypedDict):

    #current cells that are to be processed
    current_cell: CellInfo
    notebook_name:str
    notebook_path:str
    
    ##context taken from the previous cells
    previous_cells: List[CellInfo]

    #Processing_flags
    is_significant: bool
    skip_reason: Optional[str]

    #Generated code
    explanation: Optional[str]
    formatted_markdown: Optional[str]

    #File is being tracked
    docs_filepath: str
    current_docs_content: Optional[str]