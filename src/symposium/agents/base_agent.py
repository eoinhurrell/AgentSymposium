from __future__ import annotations

from typing import Any, Dict, List, Optional, TypeVar, Union, Callable
from enum import Enum
from pathlib import Path
from pydantic import BaseModel, Field
from langchain_core.language_models.llms import BaseLLM
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, END

from symposium.models.base import (
    SeverityLevel,
    CodeFile,
    CodeLocation,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
)


# Base State for our Symposium workflow
class SymposiumState(BaseModel):
    """State for the Symposium workflow"""

    messages: List[Union[AIMessage, HumanMessage, SystemMessage]] = Field(
        default_factory=list
    )
    current_agent: str = Field(default="")
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Symposium-specific fields
    pull_request: Optional[PullRequest] = None
    review_comments: List[ReviewComment] = Field(default_factory=list)

    def add_message(
        self, message: Union[AIMessage, HumanMessage, SystemMessage]
    ) -> None:
        """Add a message to the state"""
        self.messages.append(message)

    def add_review_comment(self, comment: ReviewComment) -> None:
        """Add a review comment to the state"""
        self.review_comments.append(comment)
        # If we have a pull request, also add it there
        if self.pull_request:
            self.pull_request.add_comment(comment)

    def get_comments_by_severity(self, severity: SeverityLevel) -> List[ReviewComment]:
        """Get comments by severity level"""
        return [c for c in self.review_comments if c.severity == severity]

    def get_comments_by_agent(self, agent_name: str) -> List[ReviewComment]:
        """Get comments by source agent"""
        return [c for c in self.review_comments if c.source_agent == agent_name]


class BaseAgent:
    """Base Agent class compatible with LangGraph for Symposium"""

    def __init__(
        self,
        name: str,
        llm: BaseLLM,
        system_prompt: str = "",
        config: Optional[RunnableConfig] = None,
    ):
        self.name = name
        self.llm = llm
        self.system_prompt = system_prompt
        self.config = config or {}

    def initialize_context(self, state: SymposiumState) -> SymposiumState:
        """Initialize any agent-specific context"""
        if self.system_prompt:
            state.add_message(SystemMessage(content=self.system_prompt))
        return state

    def process(self, state: SymposiumState) -> SymposiumState:
        """Process the current state and generate a response"""
        # This method should be implemented by subclasses
        state.current_agent = self.name
        return state

    def execute(self, state: SymposiumState) -> Dict[str, Any]:
        """Execute any necessary actions and return outputs"""
        result = {}
        return result

    def create_review_comment(
        self,
        file_path: str,
        line_start: int,
        message: str,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
        line_end: Optional[int] = None,
        suggestion: Optional[str] = None,
        category: Optional[str] = None,
    ) -> ReviewComment:
        """Helper method to create a standardized review comment"""
        location = CodeLocation(
            file_path=file_path, line_start=line_start, line_end=line_end
        )

        return ReviewComment(
            severity=severity,
            location=location,
            message=message,
            suggestion=suggestion,
            source_agent=self.name,
            category=category,
        )
