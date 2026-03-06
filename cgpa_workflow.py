""" We create a langraph workflow for a cgpa a student cgpa. Using both those metrics we predict if he is placed or not. We take in their cgpa projects (float, str) and then we have a thing called as priority category. There we make it as placed or not. this is going to be priority based. Now conditions here would be if cgpa is bigger than 7, and if there are projects made, then the priority is high, if not, it is low, and then finally it goes to placed. """

from langgraph.graph import StateGraph, START, END
from typing import TypedDict

class CGPAState(TypedDict):
    cgpa: float
    projects: list[str]
    priority: str
    placed: bool

def check_cgpa(state: CGPAState) -> CGPAState:
    cgpa = state['cgpa']
    projects = state['projects']

    if cgpa > 7 and len(projects) > 0:
        state['priority'] = 'High'
    else:
        state['priority'] = 'Low'

    return state

def check_placement(state : CGPAState) -> CGPAState:
    priority = state['priority']

    if priority == 'High':
        state['placed'] = True
    else:
        state['placed'] = False
    return state

graph = StateGraph(CGPAState)

graph.add_node('check_cgpa', check_cgpa)
graph.add_node('check_placement', check_placement)

graph.add_edge(START, 'check_cgpa')
graph.add_edge('check_cgpa', 'check_placement')
graph.add_edge('check_placement', END)

workflow = graph.compile()

print(workflow.invoke({"cgpa": 8.0, "projects": ["Data Extraction"]}))


