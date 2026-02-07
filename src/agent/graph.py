from langgraph.graph import StateGraph, END
from typing import Literal

from agent.state import AgentState
from agent.nodes import (
    check_significance,
    gather_context,
    generate_explanation,
    format_markdown,
    update_file,
)

def should_continue(state : AgentState) -> Literal["gather_context", "end"]:
    """
    This is our conditional edge that we run after checking for significance.
    """
    if state["is_significant"]:
        return "gather_context"
    else:
        return "end"

def create_documentation_graph() -> StateGraph:
    """
    Create and compile the documentation here in this graph. The entire workflow is here
    """
    workflow = StateGraph(AgentState)

    workflow.add_node("check_significance", check_significance)
    workflow.add_node("gather_context", gather_context)
    workflow.add_node("generate_explanation", generate_explanation)
    workflow.add_node("format_markdown", format_markdown)
    workflow.add_node("update_file", update_file)

    workflow.set_entry_point('check_significance')

    workflow.add_conditional_edges(
        'check_significance',
        should_continue,
        {
            "gather_context" : "gather_context",
            "end" : END,
        }
    )

    #Sequential edges for the main flow are here
    workflow.add_edge("gather_context", "generate_explanation")
    workflow.add_edge("generate_explanation", "format_markdown")
    workflow.add_edge("format_markdown", "update_file")
    workflow.add_edge("update_file", END)

    return workflow.compile()

_compiled_graph = None

def get_documentation_graph():
    global _compiled_graph
    if _compiled_graph is None:
        _compiled_graph = create_documentation_graph()
    return _compiled_graph