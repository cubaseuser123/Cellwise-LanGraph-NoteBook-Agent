# Documentation: bmi_workflow
> Auto-generated documentation for Jupyter notebook cells.
> Generated on: 2026-02-08 12:05:59

### Explanation
### High-Level Summary
This code snippet imports essential components from the `langgraph` library and Python's `typing` module to set up a foundation for building a state-based graph workflow. Specifically, it imports `StateGraph`, `START`, and `END` from `langgraph.graph`, along with `TypedDict` from `typing`. These imports are typically used to define and manage a state machine or workflow graph where nodes represent states or operations, and edges define transitions between them. The `TypedDict` is likely used to enforce type safety for the state data structure that will be passed through the graph.

### Detailed Breakdown
1. **Imports from `langgraph.graph`**:
   - `StateGraph`: This is a class or function that enables the creation of a state-based graph. It allows defining nodes (states) and edges (transitions) to model workflows or processes where the system moves from one state to another based on predefined logic.
   - `START` and `END`: These are likely special markers or constants used to denote the entry and exit points of the graph. `START` represents the initial state where the workflow begins, while `END` signifies the terminal state where the workflow concludes.

2. **Import from `typing`**:
   - `TypedDict`: This is a Python type hint that allows defining dictionaries with specific key-value type pairs. In the context of a `StateGraph`, `TypedDict` would be used to define the structure of the state data, ensuring that each key in the state dictionary adheres to a specified type. This enhances code clarity and helps catch type-related errors early.

### Relation to Previous Context
Since there is no previous context provided, this code appears to be the initial setup for a workflow or state machine implementation. The imports suggest that the subsequent code will involve defining a graph with states and transitions, likely using `TypedDict` to enforce a structured state format. This is a common pattern in workflow automation, where states represent different stages of a process, and transitions are triggered by specific conditions or actions. The use of `langgraph` implies that this might be part of a larger system for managing complex workflows or decision-making processes.

```python
from langgraph.graph import StateGraph, START, END
from typing import TypedDict
```
### Explanation
### High-Level Summary
This code defines a `BMIState` class that inherits from `TypedDict`, which is used to represent the state of a Body Mass Index (BMI) calculation workflow. The class specifies the structure of the state data, including the input parameters (`weight_kg` and `height_m`), the computed output (`bmi`), and a categorical classification (`category`). This is likely part of a larger system (as hinted by the previous context) where `StateGraph` from `langgraph` is used to model a workflow or pipeline for BMI calculation.

### Detailed Breakdown
1. **Class Definition and Inheritance**:
   - `BMIState` is defined as a subclass of `TypedDict`, which is a Python type hint for dictionaries with specific key-value pairs. This ensures type safety and clarity in the expected structure of the state data.
   - The keys in the `TypedDict` are:
     - `weight_kg`: A float representing weight in kilograms.
     - `height_m`: A float representing height in meters.
     - `bmi`: A float representing the computed BMI value.
     - `category`: A string representing the BMI category (e.g., "Underweight", "Normal", "Overweight").

2. **Purpose in the Workflow**:
   - This `BMIState` class is likely used as the state representation in a `StateGraph` (from the previous context), where nodes in the graph represent steps in the BMI calculation process (e.g., input validation, computation, categorization).
   - The `TypedDict` structure ensures that the state data passed between nodes in the graph adheres to a consistent schema, reducing errors and improving maintainability.

3. **Relation to Previous Context**:
   - The previous cell imported `StateGraph` and `TypedDict`, indicating that this code is part of a graph-based workflow system. The `BMIState` class will be used to define the state transitions in the graph, where each node in the graph may read or modify the state (e.g., compute `bmi` from `weight_kg` and `height_m`, or assign a `category` based on the `bmi` value).

### Technical Explanation
The `BMIState` class serves as a blueprint for the data structure that will be passed through the workflow. By using `TypedDict`, the code enforces type hints for each field, which is particularly useful in larger systems where state consistency is critical. For example, when the `StateGraph` processes the state, it can rely on the fact that `weight_kg` and `height_m` are floats, and `category` is a string. This class is a foundational component for the graph-based workflow, ensuring that all nodes in the graph operate on a well-defined and predictable state structure. The next steps in the workflow would likely involve defining nodes that compute the BMI or classify it into categories, all while adhering to this state structure.

```python
class BMIState(TypedDict):
    weight_kg : float
    height_m : float
    bmi: float
    category: str
```
### Explanation
### High-Level Summary
This code defines a function `calculate_BMI` that computes the Body Mass Index (BMI) from a given state dictionary containing weight (in kilograms) and height (in meters). The function updates the state with the calculated BMI value (rounded to 2 decimal places) and returns the modified state. It is part of a larger workflow (likely a graph-based system, given the previous context) where BMI calculations are performed as a step in processing health-related data.

### Detailed Breakdown
1. **Function Signature**:
   - The function takes a single parameter `state` of type `BMIState` (a `TypedDict` with keys `weight_kg`, `height_m`, `bmi`, and `category`).
   - It returns the same `BMIState` object after updating the `bmi` field.

2. **Key Operations**:
   - **Input Extraction**: The function retrieves `weight_kg` and `height_m` from the input state.
   - **BMI Calculation**: Computes BMI using the formula `weight_kg / (height_m**2)`.
   - **State Update**: Stores the rounded BMI value (to 2 decimal places) in the `bmi` field of the state.
   - **Return**: Returns the updated state, preserving all other fields (e.g., `category`).

3. **Relation to Previous Context**:
   - The function aligns with the `BMIState` `TypedDict` defined earlier, ensuring type consistency in the workflow.
   - It is likely designed as a node in a `StateGraph` (from `langgraph.graph`), where each node processes a specific part of the state. The `START` and `END` markers (imported earlier) suggest this is part of a directed graph for sequential or conditional state transformations.

### Technical Explanation
The function adheres to a functional programming style, where the input state is treated as immutable (though Python dictionaries are mutable, the function avoids side effects by explicitly returning the modified state). The BMI calculation follows the standard medical formula, and rounding ensures readability of the result. By returning the entire state, the function enables chaining with other graph nodes (e.g., a subsequent node might populate the `category` field based on the computed `bmi`). The use of `TypedDict` enforces structural consistency, which is critical for graph-based workflows where state integrity must be maintained across nodes.

```python
def calculate_BMI(state : BMIState) -> BMIState:
    weight_kg = state['weight_kg']
    height_m = state['height_m']
    bmi = weight_kg / (height_m**2)
    state['bmi'] = round(bmi, 2)

    return state
```
### Explanation
### High-Level Summary
The provided Python code defines a function `label_bmi` that categorizes a calculated BMI (Body Mass Index) value into one of four health-related categories: "Underweight," "Normal weight," "Overweight," or "Obesity." This function operates on a `BMIState` dictionary (a `TypedDict` defined in the previous context) and updates it with the appropriate category based on the BMI value stored in the state. The function is designed to work as part of a workflow where BMI is first calculated (as seen in `calculate_BMI` from Cell [5]) and then labeled for interpretation.

### Detailed Breakdown
1. **Function Signature and Input**:
   - The function `label_bmi` takes a single parameter `state` of type `BMIState`, which is a `TypedDict` containing keys `weight_kg`, `height_m`, `bmi`, and `category`. The `state` is expected to already contain a computed `bmi` value (from `calculate_BMI`).
   - The function does not return any new data structure but instead modifies the input `state` in-place by updating the `category` field.

2. **Logic and Conditions**:
   - The function retrieves the `bmi` value from the `state` dictionary.
   - It then checks the `bmi` against standard health thresholds:
     - **Underweight**: BMI < 18.5
     - **Normal weight**: 18.5 ≤ BMI < 24.9
     - **Overweight**: 25 ≤ BMI < 29.9
     - **Obesity**: BMI ≥ 30
   - Based on the condition met, the `category` field in `state` is updated with the corresponding string label.

3. **Return Value**:
   - The function returns the modified `state` dictionary, now including the `category` label. This allows the function to be chained in a workflow (e.g., in a `StateGraph` from `langgraph`, as hinted by the imports in Cell [3]).

### Relation to Previous Context
This function is part of a multi-step workflow for BMI calculation and categorization. The `BMIState` `TypedDict` (Cell [4]) defines the structure of the data being processed, while `calculate_BMI` (Cell [5]) computes the BMI value from `weight_kg` and `height_m`. The `label_bmi` function is the next logical step, taking the computed BMI and assigning a human-readable category. Together, these functions could be nodes in a graph-based workflow (e.g., using `StateGraph`), where data flows from calculation to labeling. The modular design ensures separation of concerns: calculation and categorization are distinct steps, making the code easier to maintain and extend.

```python
def label_bmi(state: BMIState) -> BMIState:
    bmi = state['bmi']
    if bmi < 18.5:
        state['category'] = "Underweight"
    elif 18.5 <= bmi < 24.9:
        state['category'] = "Normal weight"
    elif 25 <= bmi < 29.9:
        state['category'] = "Overweight"
    else:
        state['category'] = "Obesity"
    
    return state
```
### Explanation
### High-Level Summary
The current cell initializes a `StateGraph` object from the `langgraph.graph` module, using the `BMIState` `TypedDict` as its state schema. This sets up a graph-based workflow where nodes represent operations (e.g., functions like `calculate_BMI` and `label_bmi`) and edges define the execution flow. The `StateGraph` will manage the state transitions, ensuring the `BMIState` dictionary is passed and updated correctly between nodes.

### Detailed Breakdown
1. **`StateGraph(BMIState)`**:
   - **Purpose**: Creates a directed graph where each node processes a state of type `BMIState`. The graph will orchestrate the execution of functions (nodes) that modify the state, such as calculating BMI or assigning a category.
   - **Parameters**:
     - `BMIState`: A `TypedDict` defining the structure of the state object, which includes `weight_kg`, `height_m`, `bmi`, and `category`. This ensures type consistency across all nodes in the graph.
   - **Return Value**: An instance of `StateGraph` configured to work with `BMIState`. This instance will later be used to add nodes (e.g., `calculate_BMI`, `label_bmi`) and define edges between them.

2. **Relation to Previous Context**:
   - The `BMIState` class (Cell [4]) and the functions `calculate_BMI` (Cell [5]) and `label_bmi` (Cell [6]) are prerequisites for this graph. The graph will use these functions as nodes to process the state. For example:
     - `calculate_BMI` will compute the BMI from `weight_kg` and `height_m`.
     - `label_bmi` will categorize the BMI into a health category.
   - The graph’s initialization here is the first step in building a pipeline that chains these operations, likely starting from `START` (input state) and ending at `END` (final categorized state).

### Technical Explanation
The `StateGraph` is a framework for defining workflows where state mutations are explicit and sequential. By passing `BMIState` as the schema, the graph enforces that all nodes must accept and return a dictionary conforming to this structure. This ensures type safety and clarity in state management. In subsequent steps, nodes like `calculate_BMI` and `label_bmi` will be added to the graph, and edges will define their execution order (e.g., `START -> calculate_BMI -> label_bmi -> END`). The graph’s design aligns with the previous functions, which are stateless and purely transformative, making them ideal for graph-based orchestration.

```python
graph = StateGraph(BMIState)
```
### Explanation
### High-Level Summary
This code constructs a directed workflow graph for calculating and categorizing Body Mass Index (BMI) using the `StateGraph` framework from `langgraph`. The graph orchestrates two sequential operations: first computing the BMI from weight and height inputs, then classifying the result into a health category. The structure ensures data flows linearly from input to output, with each node representing a discrete processing step.

### Detailed Breakdown
The code begins by registering two nodes (`calculate_bmi` and `label_bmi`) in the graph, each bound to their respective functions from previous cells. The `add_node` method associates a string identifier with a function that processes the `BMIState` TypedDict. The edges define the execution order:
1. `START → calculate_bmi`: Initiates the workflow by passing the initial state (containing `weight_kg` and `height_m`) to the BMI calculation.
2. `calculate_bmi → label_bmi`: Routes the updated state (now including the computed `bmi`) to the categorization function.
3. `label_bmi → END`: Terminates the workflow after assigning a `category` based on BMI thresholds.

The graph implicitly handles state propagation between nodes, where each function modifies and returns the state dictionary. The `TypedDict` structure ensures type consistency across operations, while the linear edge configuration enforces a strict dependency chain—categorization cannot occur without prior BMI calculation.

### Relation to Previous Context
This cell builds directly upon Cells [3]–[8], which defined the state schema (`BMIState`), core functions (`calculate_BMI`, `label_bmi`), and initialized the empty graph. The current code *assembles* these components into an executable pipeline, transforming the standalone functions into an interconnected system. The graph’s design mirrors the logical flow of BMI assessment: raw inputs → computation → interpretation, with the `StateGraph` framework abstracting away manual state management between steps. The result is a reusable, declarative workflow that can be invoked with any valid `BMIState` input.

```python
graph.add_node('calculate_bmi', calculate_BMI)
graph.add_node('label_bmi', label_bmi)

graph.add_edge(START, 'calculate_bmi')
graph.add_edge('calculate_bmi', "label_bmi")
graph.add_edge('label_bmi', END)
```
### Explanation
### High-Level Summary
The current cell (`workflow = graph.compile()`) compiles the previously defined `StateGraph` (from Cell [8]) into an executable workflow. This workflow orchestrates the sequential execution of the BMI calculation and categorization logic defined in Cells [5] and [6], respectively. The compilation step transforms the graph's structure—nodes (functions) and edges (dependencies)—into an optimized, runnable pipeline that processes `BMIState` objects.

### Detailed Breakdown
1. **Compilation Process**:
   - The `compile()` method analyzes the graph's topology (nodes: `'calculate_bmi'` and `'label_bmi'`, edges: `START → calculate_bmi → label_bmi → END`) to generate an efficient execution plan.
   - It ensures dependencies are respected (e.g., `label_bmi` cannot run before `calculate_bmi` updates the `bmi` field in the state).
   - The resulting `workflow` is a callable object that, when invoked with an initial `BMIState`, will automatically traverse the graph, applying each node's function in the specified order.

2. **Key Operations**:
   - **State Propagation**: The workflow passes the `BMIState` (a `TypedDict` with fields `weight_kg`, `height_m`, `bmi`, and `category`) between nodes, mutating it incrementally.
   - **Edge Handling**: The compiled workflow embeds the control flow logic (e.g., skipping nodes if edges are conditional, though this example uses linear edges).
   - **Optimization**: The compiler may optimize memory usage or parallelize independent branches (not applicable here due to the linear graph).

3. **Parameters and Return Values**:
   - **Input**: The `workflow` expects an initial `BMIState` with at least `weight_kg` and `height_m` populated (other fields may be `None` or missing).
   - **Output**: Returns the same `BMIState` object, now fully populated with `bmi` (calculated) and `category` (labeled), e.g.:
     ```python
     {"weight_kg": 70, "height_m": 1.75, "bmi": 22.86, "category": "Normal weight"}
     ```

### Relation to Previous Context
This cell is the culmination of the graph construction started in Cell [8]. The `StateGraph` was assembled by:
1. Defining the state schema (`BMIState` in Cell [4]).
2. Implementing node functions (`calculate_BMI` in Cell [5] and `label_bmi` in Cell [6]).
3. Adding nodes and edges to the graph (Cell [10]), specifying the execution order.
The `compile()` call now converts this declarative graph into an imperative workflow, ready for execution. For example, invoking `workflow.invoke({"weight_kg": 70, "height_m": 1.75})` would trigger the entire BMI calculation and categorization pipeline. This pattern is common in stateful workflow systems (e.g., LangGraph), where graphs model complex processes as composable steps.

```python
workflow = graph.compile()
```
### Explanation
### High-Level Summary
This code executes a predefined workflow to calculate the Body Mass Index (BMI) and categorize it based on standard health guidelines. The workflow is structured as a state machine where an initial state (containing weight and height) is processed through a series of nodes (`calculate_bmi` and `label_bmi`) to produce a final state that includes the computed BMI and its corresponding health category. The execution is triggered by invoking the compiled workflow with the initial state, and the result is printed.

### Detailed Breakdown
1. **Initial State Setup**:
   The `initial_state` dictionary is initialized with two keys: `weight_kg` (80 kg) and `height_m` (1.73 meters). This dictionary adheres to the `BMIState` type (implied by the context), which is the input schema for the workflow.

2. **Workflow Invocation**:
   The `workflow.invoke(initial_state)` call triggers the state machine execution. The workflow follows the graph structure defined earlier:
   - **Step 1**: The `calculate_bmi` node (from Cell [5]) computes the BMI using the formula `weight_kg / (height_m ** 2)` and rounds it to 2 decimal places, storing the result in the state under the key `bmi`.
   - **Step 2**: The `label_bmi` node (from Cell [6]) categorizes the BMI into one of four health categories ("Underweight," "Normal weight," "Overweight," or "Obesity") based on conditional checks and adds this label to the state under the key `category`.

3. **Output**:
   The `final_state` contains the original inputs (`weight_kg`, `height_m`), the computed `bmi` (e.g., `26.7` for the given inputs), and the `category` (e.g., "Overweight"). This is printed to the console.

### Relation to Previous Context
This cell builds directly on the workflow defined in Cells [8]–[11], where:
- A `StateGraph` was created to model the BMI calculation pipeline.
- Nodes (`calculate_bMI` and `label_bmi`) were added to the graph, along with edges defining their execution order.
- The graph was compiled into an executable `workflow`.

The current cell demonstrates the practical use of this workflow by providing an initial state and executing the pipeline. The output validates the correctness of the graph structure and the individual functions, showing how state is transformed step-by-step. This is a classic example of a functional pipeline where data flows through pure functions (no side effects) to produce a deterministic result.

```python
initial_state = {"weight_kg": 80, "height_m":1.73}

final_state = workflow.invoke(initial_state)

print(final_state)
```
### Explanation
### High-Level Summary
This Python code defines a state-based workflow using the `langgraph` library to evaluate a student's eligibility for placement based on their CGPA (Cumulative Grade Point Average) and project experience. The workflow consists of two nodes: `check_cgpa` and `check_placement`, which sequentially process the input data to determine a priority level (`High` or `Low`) and a placement status (`True` or `False`). The graph structure ensures the nodes execute in a predefined order, and the final result is printed for a sample input.

### Detailed Breakdown
1. **State Definition (`CGPAState`)**:
   - A `TypedDict` named `CGPAState` is defined to structure the data flowing through the graph. It includes:
     - `cgpa` (float): The student's CGPA.
     - `projects` (list[str]): A list of project names the student has worked on.
     - `priority` (str): A field to store the priority level (`High` or `Low`), initialized implicitly during processing.
     - `placed` (bool): A field to store the placement status, also initialized during processing.

2. **Node Functions**:
   - `check_cgpa(state: CGPAState) -> CGPAState`:
     - Evaluates the `cgpa` and `projects` fields. If the CGPA is greater than 7 **and** the student has at least one project, the `priority` is set to `'High'`; otherwise, it is set to `'Low'`.
     - Returns the updated `state`.
   - `check_placement(state: CGPAState) -> CGPAState`:
     - Checks the `priority` field. If `'High'`, the `placed` field is set to `True`; otherwise, it is set to `False`.
     - Returns the updated `state`.

3. **Graph Construction**:
   - A `StateGraph` is initialized with `CGPAState` as the state schema.
   - Nodes (`check_cgpa` and `check_placement`) are added to the graph, each mapped to their respective functions.
   - Edges are defined to enforce the execution order: `START -> check_cgpa -> check_placement -> END`.
   - The graph is compiled into an executable `workflow`.

4. **Execution**:
   - The workflow is invoked with a sample input: `{"cgpa": 8.0, "projects": ["Data Extraction"]}`.
   - The output will reflect the processed state, with `priority` set to `'High'` and `placed` set to `True` due to the input meeting the criteria.

### Relation to Previous Context
Since there is no previous context, this code stands alone as a self-contained example of a stateful workflow using `langgraph`. It demonstrates how to model conditional logic and sequential processing in a graph-based system, which could be extended for more complex decision-making pipelines (e.g., multi-stage evaluations or branching paths). The use of `TypedDict` ensures type safety and clarity in the state structure.

```python
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
```
