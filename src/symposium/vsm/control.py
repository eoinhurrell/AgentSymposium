# ======= VSM SYSTEM 3: CONTROL =======
# Manages the overall review process and resource allocation
import re
import uuid
from typing import Any, Dict, List, Optional, Union

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama

# LangGraph imports
from langgraph.graph import StateGraph, END

from symposium.models.base import (
    SeverityLevel,
    CodeFile,
    FileMetadata,
    CodeLocation,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
    CodeReviewState,
)


def system3_control_review_process(
    state: CodeReviewState,
) -> Union[CodeReviewState, str]:
    """
    VSM System 3: Control
    Controls the review process, determines what to check next, and when to finish.
    """
    print("CONTROL")
    new_state = state.model_copy(deep=True)

    # Check if we have a pull request to review
    if not new_state.pull_request:
        new_state.messages.append(
            AIMessage(content="No pull request provided for review.")
        )
        return new_state

    # Check which review components have been run
    components_run = new_state.components_run

    # Determine the next component to run based on a predefined sequence
    if "linter" not in components_run:
        new_state.components_run = components_run.union({"linter"})
        new_state.current_agent = "linter"
        return new_state

    if "security_checker" not in components_run:
        new_state.components_run = components_run.union({"security_checker"})
        new_state.current_agent = "security_checker"
        return new_state

    if "complexity_analyzer" not in components_run:
        new_state.components_run = components_run.union({"complexity_analyzer"})
        new_state.current_agent = "complexity_analyzer"
        return new_state

    if "coordinator" not in components_run:
        new_state.components_run = components_run.union({"coordinator"})
        new_state.current_agent = "coordinator"
        return new_state

    if "intelligence" not in components_run:
        new_state.components_run = components_run.union({"intelligence"})
        new_state.current_agent = "intelligence"
        return new_state

    if "policy" not in components_run:
        new_state.components_run = components_run.union({"policy"})
        new_state.current_agent = "policy"
        return new_state

    # If all components have been run, we're done
    new_state.messages.append(AIMessage(content="Code review completed."))
    print("DONE")
    new_state.current_agent = ""
    return new_state
