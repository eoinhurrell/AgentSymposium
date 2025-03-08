"""
Core data models for the VSM-based code review system.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Union, Any, Set

from pydantic import BaseModel, Field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage


class SeverityLevel(str, Enum):
    """Severity levels for review comments."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    INFO = "info"


class CodeLocation(BaseModel):
    """Represents a location in a code file."""

    file_path: str
    line_start: int
    line_end: Optional[int] = None
    column_start: Optional[int] = None
    column_end: Optional[int] = None

    def __str__(self) -> str:
        """String representation of the location."""
        if self.line_end and self.line_end != self.line_start:
            return f"{self.file_path}:{self.line_start}-{self.line_end}"
        return f"{self.file_path}:{self.line_start}"


class ReviewComment(BaseModel):
    """Model for storing code review comments."""

    id: Optional[str] = None
    severity: SeverityLevel = SeverityLevel.MEDIUM
    location: CodeLocation
    message: str
    suggestion: Optional[str] = None
    context: Optional[str] = None

    # For tracking the agent that generated this comment
    source_agent: Optional[str] = None

    # For grouping related comments
    category: Optional[str] = None
    related_comment_ids: List[str] = Field(default_factory=list)

    def __str__(self) -> str:
        """String representation of the review comment."""
        result = f"[{self.severity.upper()}] {self.location}: {self.message}"
        if self.suggestion:
            result += f"\nSuggestion: {self.suggestion}"
        return result


class ReviewResult(BaseModel):
    comments: List[ReviewComment] = Field(default_factory=list)


class FileMetadata(BaseModel):
    """Metadata for a code file."""

    path: str
    language: str
    is_new: bool = False
    is_deleted: bool = False
    is_renamed: bool = False
    old_path: Optional[str] = None
    line_count: Optional[int] = None

    @property
    def file_extension(self) -> str:
        """Get the file extension."""
        return Path(self.path).suffix


class CodeFile(BaseModel):
    """Model representing a file to be reviewed."""

    content: str
    metadata: FileMetadata
    diff: Optional[str] = None

    def get_lines(self) -> List[str]:
        """Get the content as a list of lines."""
        return self.content.splitlines()

    def get_line(self, line_number: int) -> str:
        """Get a specific line by line number (1-indexed)."""
        lines = self.get_lines()
        if 1 <= line_number <= len(lines):
            return lines[line_number - 1]
        raise IndexError(f"Line number {line_number} is out of range")


class PullRequestMetadata(BaseModel):
    """Metadata for a pull request."""

    id: str
    title: str
    description: Optional[str] = None
    author: str
    base_branch: str
    head_branch: str
    created_at: str
    updated_at: str


class PullRequest(BaseModel):
    """Model representing a pull request."""

    metadata: PullRequestMetadata
    files: List[CodeFile] = Field(default_factory=list)
    comments: List[ReviewComment] = Field(default_factory=list)

    def add_file(self, file: CodeFile) -> None:
        """Add a file to the PR."""
        self.files.append(file)

    def add_comment(self, comment: ReviewComment) -> None:
        """Add a comment to the PR."""
        self.comments.append(comment)

    def get_file_by_path(self, path: str) -> Optional[CodeFile]:
        """Get a file by its path."""
        for file in self.files:
            if file.metadata.path == path:
                return file
        return None

    def get_comments_by_file(self, path: str) -> List[ReviewComment]:
        """Get all comments for a specific file."""
        return [c for c in self.comments if c.location.file_path == path]

    def get_comments_by_severity(self, severity: SeverityLevel) -> List[ReviewComment]:
        """Get all comments with a specific severity."""
        return [c for c in self.comments if c.severity == severity]


class CodeReviewState(BaseModel):
    """State for the code review agent."""

    messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = Field(
        default_factory=list
    )
    current_agent: str = Field(default="")
    components_run: Set[str] = Field(default_factory=set)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    context: Dict[str, Any] = Field(default_factory=dict)
    outputs: Dict[str, Any] = Field(default_factory=dict)

    # Code review specific fields
    pull_request: Optional[PullRequest] = None
