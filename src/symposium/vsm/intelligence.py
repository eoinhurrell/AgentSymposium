# ======= VSM SYSTEM 4: INTELLIGENCE =======
# Analyzes the broader context and environment
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


def system4_analyze_context(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 4: Intelligence
    Analyzes the broader context of the code changes, including PR metadata,
    repository trends, and development context.
    """
    print("ANALYZE")
    new_state = state.model_copy(deep=True)

    if not new_state.pull_request:
        return new_state

    # Analyze pull request metadata for context
    pr_metadata = new_state.pull_request.metadata

    # Check if the PR description is informative
    if not pr_metadata.description or len(pr_metadata.description) < 10:
        new_state.context["pr_description_quality"] = "poor"
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.LOW,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message="Pull request description is missing or too brief.",
            suggestion="Add a more detailed description explaining the purpose and context of these changes.",
            source_agent="intelligence",
            category="documentation",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["pr_description_quality"] = "good"

    # Analyze code changes as a whole
    total_lines_changed = sum(
        len(file.content.split("\n")) for file in new_state.pull_request.files
    )
    if total_lines_changed > 500:
        new_state.context["pr_size"] = "large"
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.MEDIUM,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message=f"This pull request is quite large ({total_lines_changed} lines).",
            suggestion="Consider breaking this into smaller, more focused pull requests for easier review.",
            source_agent="intelligence",
            category="process",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["pr_size"] = "reasonable"

    # Check if the PR affects critical components
    critical_paths = ["security", "auth", "payment", "core"]
    affected_critical_paths = []

    for file in new_state.pull_request.files:
        for path in critical_paths:
            if path in file.metadata.path.lower():
                affected_critical_paths.append(path)

    if affected_critical_paths:
        new_state.context["affects_critical_components"] = True
        paths_str = ", ".join(set(affected_critical_paths))
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.HIGH,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message=f"This PR affects critical components: {paths_str}",
            suggestion="Ensure thorough testing and consider requesting additional reviewers with expertise in these areas.",
            source_agent="intelligence",
            category="risk",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["affects_critical_components"] = False

    # Add a message about the context analysis
    files_count = len(new_state.pull_request.files)
    new_state.messages.append(
        AIMessage(
            content=f"Analyzed the context of the pull request with {files_count} files and {total_lines_changed} lines changed."
        )
    )

    return new_state
