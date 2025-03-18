import pytest
from unittest.mock import MagicMock, patch
import re
import uuid

from langchain_core.messages import AIMessage, SystemMessage, HumanMessage
from pydantic import BaseModel, Field
from typing import List

from symposium.models.base import (
    SeverityLevel,
    CodeLocation,
    ReviewComment,
    CodeReviewState,
    PullRequest,
    CodeFile,
    FileMetadata,
)

# Import the function to test - update the module path as needed
from symposium.systems.operational import system1_lint_code


# Define the LintingResult class used in the function
class LintingResult(BaseModel):
    comments: List[ReviewComment] = Field(default_factory=list)


@pytest.fixture
def mock_state():
    """Create a mock state with a pull request and files for testing."""
    state = CodeReviewState(
        pull_request=PullRequest(
            id="pr-123",
            title="Test PR",
            description="Test pull request for linting",
            comments=[],
            files=[
                CodeFile(
                    metadata=FileMetadata(path="test_file.py", language="python"),
                    content="def test_function():\n    pass\n",
                )
            ],
        ),
        messages=[],
    )
    return state


def test_system1_lint_code_normal(mock_state, monkeypatch):
    """Test normal operation of system1_lint_code with successful linting."""
    # Set up mock chain
    mock_llm = MagicMock()
    mock_with_tools = MagicMock()
    mock_with_structured_output = MagicMock()

    # Set up the mock chain returns
    mock_llm.bind_tools.return_value = mock_with_tools
    mock_with_tools.with_structured_output.return_value = mock_with_structured_output

    # Create mock linting results
    mock_comment = ReviewComment(
        id="",  # Will be replaced with a new UUID
        severity=SeverityLevel.MEDIUM,
        location=CodeLocation(
            file_path="", line_start=2, line_end=2  # Will be updated in the function
        ),
        message="Missing docstring",
        suggestion="Add a docstring to describe the function",
        source_agent="linter_llm",
        category="style",
    )

    mock_result = LintingResult(comments=[mock_comment])
    mock_with_structured_output.invoke.return_value = mock_result

    # Patch the necessary imports
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda *args, **kwargs: mock_llm)
    monkeypatch.setattr("tools.run_linter", MagicMock())

    # Run the function
    result_state = system1_lint_code(mock_state)

    # Assertions
    assert len(result_state.pull_request.comments) == 1

    # Check comment content
    comment = result_state.pull_request.comments[0]
    assert comment.message == "Missing docstring"
    assert comment.location.file_path == "test_file.py"
    assert comment.location.line_start == 2
    assert comment.suggestion == "Add a docstring to describe the function"

    # Verify UUID was assigned
    assert comment.id != ""
    assert re.match(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", comment.id
    )

    # Check summary message
    assert len(result_state.messages) == 1
    assert (
        "Performed code style analysis and found 1 style issues"
        in result_state.messages[0].content
    )


def test_system1_lint_code_error(mock_state, monkeypatch):
    """Test error handling in system1_lint_code."""
    # Set up mock chain
    mock_llm = MagicMock()
    mock_with_tools = MagicMock()
    mock_with_structured_output = MagicMock()

    # Set up the mock chain returns
    mock_llm.bind_tools.return_value = mock_with_tools
    mock_with_tools.with_structured_output.return_value = mock_with_structured_output

    # Make the LLM call raise an exception
    mock_with_structured_output.invoke.side_effect = Exception("Test error")

    # Patch the necessary imports
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda *args, **kwargs: mock_llm)
    monkeypatch.setattr("tools.run_linter", MagicMock())

    # Run the function
    result_state = system1_lint_code(mock_state)

    # Assertions
    assert len(result_state.pull_request.comments) == 1

    # Check error comment content
    comment = result_state.pull_request.comments[0]
    assert "Error during style analysis" in comment.message
    assert "Test error" in comment.message
    assert comment.severity == SeverityLevel.LOW
    assert comment.source_agent == "linter_error"

    # Check summary message
    assert len(result_state.messages) == 1
    assert "Performed code style analysis" in result_state.messages[0].content


def test_system1_lint_code_no_pull_request(monkeypatch):
    """Test behavior when no pull request is present."""
    # Create state with no pull request
    state = CodeReviewState(pull_request=None, messages=[])

    # Mock the imports we want to ensure are not called
    mock_chat_ollama = MagicMock()
    mock_run_linter = MagicMock()

    monkeypatch.setattr("langchain_ollama.ChatOllama", mock_chat_ollama)
    monkeypatch.setattr("tools.run_linter", mock_run_linter)

    # Run the function
    result_state = system1_lint_code(state)

    # Assertions
    assert result_state.pull_request is None
    assert len(result_state.messages) == 0

    # Verify the mocks were not called
    assert mock_chat_ollama.call_count == 0
    assert mock_run_linter.call_count == 0


def test_system1_lint_code_message_content(mock_state, monkeypatch):
    """Test that the correct messages are passed to the LLM."""
    # Set up mock chain
    mock_llm = MagicMock()
    mock_with_tools = MagicMock()
    mock_with_structured_output = MagicMock()

    # Set up the mock chain returns
    mock_llm.bind_tools.return_value = mock_with_tools
    mock_with_tools.with_structured_output.return_value = mock_with_structured_output

    # Create empty result
    mock_result = LintingResult(comments=[])
    mock_with_structured_output.invoke.return_value = mock_result

    # Patch the necessary imports
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda *args, **kwargs: mock_llm)
    monkeypatch.setattr("tools.run_linter", MagicMock())

    # Run the function
    system1_lint_code(mock_state)

    # Capture the arguments passed to invoke
    args, kwargs = mock_with_structured_output.invoke.call_args

    # Check message structure and content
    messages = args[0]
    assert len(messages) == 2
    assert isinstance(messages[0], SystemMessage)
    assert isinstance(messages[1], HumanMessage)

    # Check system message content
    system_content = messages[0].content
    assert "expert code linter" in system_content
    assert "ReviewComment" in system_content
    assert "severity" in system_content
    assert "location" in system_content

    # Check user message content
    user_content = messages[1].content
    assert "test_file.py" in user_content
    assert "python" in user_content
    assert "def test_function():" in user_content
    assert "run_linter tool" in user_content


def test_system1_lint_code_multiple_files(monkeypatch):
    """Test handling multiple files in the pull request."""
    # Create a state with multiple files
    state = CodeReviewState(
        pull_request=PullRequest(
            id="pr-multi",
            title="Multi-file PR",
            comments=[],
            files=[
                File(
                    metadata=FileMetadata(path="file1.py", language="python"),
                    content="def func1():\n    pass\n",
                ),
                File(
                    metadata=FileMetadata(path="file2.py", language="python"),
                    content="def func2():\n    return True\n",
                ),
            ],
        ),
        messages=[],
    )

    # Set up mock chain
    mock_llm = MagicMock()
    mock_with_tools = MagicMock()
    mock_with_structured_output = MagicMock()

    # Set up the mock chain returns
    mock_llm.bind_tools.return_value = mock_with_tools
    mock_with_tools.with_structured_output.return_value = mock_with_structured_output

    # Create different responses for each file
    def invoke_side_effect(messages):
        content = messages[1].content
        if "file1.py" in content:
            return LintingResult(
                comments=[
                    ReviewComment(
                        id="",
                        severity=SeverityLevel.MEDIUM,
                        location=CodeLocation(file_path="", line_start=1),
                        message="Missing docstring in file1",
                        source_agent="linter_llm",
                    )
                ]
            )
        elif "file2.py" in content:
            return LintingResult(
                comments=[
                    ReviewComment(
                        id="",
                        severity=SeverityLevel.LOW,
                        location=CodeLocation(file_path="", line_start=2),
                        message="Explicit return not needed",
                        source_agent="linter_llm",
                    )
                ]
            )
        return LintingResult(comments=[])

    mock_with_structured_output.invoke.side_effect = invoke_side_effect

    # Patch the necessary imports
    monkeypatch.setattr("langchain_ollama.ChatOllama", lambda *args, **kwargs: mock_llm)
    monkeypatch.setattr("tools.run_linter", MagicMock())

    # Run the function
    result_state = system1_lint_code(state)

    # Assertions
    assert len(result_state.pull_request.comments) == 2
    assert mock_with_structured_output.invoke.call_count == 2

    # Verify comments for each file
    file1_comments = [
        c
        for c in result_state.pull_request.comments
        if c.location.file_path == "file1.py"
    ]
    file2_comments = [
        c
        for c in result_state.pull_request.comments
        if c.location.file_path == "file2.py"
    ]

    assert len(file1_comments) == 1
    assert "Missing docstring" in file1_comments[0].message

    assert len(file2_comments) == 1
    assert "Explicit return not needed" in file2_comments[0].message
