import asyncio 
from openai import AsyncOpenAI
from typing import Dict, Any

from ..config import get_settings
from .state import AgentState
from ..utils.significance import is_significant_cell
from ..utils.context import format_context_for_prompt,
from ..utils.file_operations import append_to_docs, read_current_docs

_client: AysncOpenAI = None

def _get_client() -> AsyncOpenAI:
    #we initialize OpenAI client for Vercel Ai Gateway
    global _client
    if _client is None:
        settings = get_settings()
        _client = AsyncOpenAI(
            base_url=settings.vercel_ai_gateway_url,
            api_key=settings.vercel_ai_gateway_api_key,
        )
    return _client

def check_significance(state: AgentState) -> Dict[str, Any]:
    """
    Remove anything that is not significant here
    """
    code = state["current_cell"]["code"]
    is_significant, skip_reason = is_significant_cell(code)
    return {"is_significant": is_significant, "skip_reason": skip_reason}

def gather_context(state: AgentState) -> Dict[str, Any]:
    """
    Here we read the current docs file content as context
    """
    docs_filepath = state["docs_filepath"]
    current_content = read_current_docs(docs_filepath)
    return {"current_docs_content" : current_content}

async def generate_explanation(state: AgentState) -> Dict[str, Any]:
    """
    Actual stuff that needs to generated happens over here
    """
    settings = get_settings()
    client = _get_client()

    current_code = state["current_cell"]["code"]
    context = format_context_for_prompt(state["previous_cells"])

    if len(current_code) > settings.max_cell_length:
        current_code = current_code[:settings.max_cell_length] + "\n... (truncated)"

    system_prompt = """
    You are a code documentation expert. Explain what the given Python code does in clear, concise terms (2-4 sentences). Focus on what it accomplishes, key operations, and how it relates to previous context.
    """
    user_prompt = f"""## Previous Context
    {context if context else "(No previous context)"}

    ##Current Cell (Execution #{state['current_cell']['execution_count']})
    ```python
    {current_code}
    ```
    Explain what this code does:"""

    try:
        response = await client.chat.completions.create(
            model = settings.model_name,
            messages=[
                {"role": "system", "contet" : system_prompt},
                {"role": "user", "content" : user_prompt},
            ],
            temperature=settings.model_temprature, 
            max_tokens=settings.model_max_tokens,
        )
        explanation = response.choices[0].message.content.strip()
    except Exception as e:
        explanation = f"*Documentation generation failed: {str(e)}*"

    return{"explanation" : explanation}

def format_markdown(state : AgentState) -> Dict[str, Any]:
    """
    Here we propely structure what we just generated into the markdown we are going for
    """
    cell = state["current_cell"]
    markdown = f"""
    ```
    ##Cell [{cell['execution_count']}]

    ```python
    {cell['code']}
    ```
    ###Explanation

    {state['explanation']}

    """
    return{"formatted_markdown" : markdown}

def update_file(state : AgentState) -> Dict[str, Any]:
    """
    Now we append the md file with new code, that part happens over here.
    """
    try:
        append_to_docs(state["docs_filepath"], state["formatted_markdown"])
    except Exception as e:
        print(f"[NotebookDocs] Error writing docs: {e}")
    return {}

def generate_explanation_sync(state : AgentState) -> Dict[str, Any]:
    """
    This will be a sync wrapper for our generate_exaplanation function. I hear its required by Langraph
    """
    return asyncio.run(generate_explanation(state))
