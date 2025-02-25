"""
Test fixtures for the VSM-based code review system.
"""

import pytest

from symposium.models.base import (
    CodeFile,
    FileMetadata,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
    CodeLocation,
    SeverityLevel,
)


@pytest.fixture
def valid_python_code() -> str:
    """Return a valid Python code snippet."""
    return """
def factorial(n):
    # Calculate the factorial of n.
    if n <= 1:
        return 1
    return n * factorial(n - 1)

def fibonacci(n):
    # Return the nth Fibonacci number.
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
"""


@pytest.fixture
def python_code_with_syntax_error() -> str:
    """Return a Python code snippet with syntax errors."""
    return '''
def factorial(n):
    """Calculate the factorial of n."""
    if n <= 1:
        return 1
    return n * factorial(n - 1

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0:
        return 0
    elif n == 1:
        return 1
    else:
        a, b = 0, 1
        for _ in range(2, n + 1):
            a, b = b, a + b
        return b
'''


@pytest.fixture
def python_code_with_style_issues() -> str:
    """Return a Python code snippet with style issues."""
    return '''
def factorial( n ):
    """Calculate the factorial of n."""
    if n<=1: return 1
    return n*factorial(n-1)

def fibonacci(n):
    """Return the nth Fibonacci number."""
    if n <= 0: return 0
    elif n == 1: return 1
    else:
        a,b = 0,1
        for _ in range(2, n + 1):
         a, b = b, a + b
        return b

def unused_function():
    pass
'''


@pytest.fixture
def python_code_with_quality_issues() -> str:
    """Return a Python code snippet with code quality issues."""
    return """
def f(n):
    if n <= 1:
        return 1
    return n * f(n - 1)

def g(n):
    x = 0
    y = 1
    z = 0
    if n <= 0:
        return x
    if n == 1:
        return y
    i = 2
    while i <= n:
        z = x + y
        x = y
        y = z
        i += 1
    return z

# Global variable
counter = 0

def increment():
    global counter
    counter += 1
    return counter
"""


@pytest.fixture
def code_file_factory():
    """Factory fixture to create CodeFile instances."""

    def _create_code_file(
        content: str,
        path: str = "test_file.py",
        language: str = "python",
        is_new: bool = False,
    ) -> CodeFile:
        metadata = FileMetadata(
            path=path,
            language=language,
            is_new=is_new,
            line_count=len(content.splitlines()),
        )
        return CodeFile(content=content, metadata=metadata)

    return _create_code_file


@pytest.fixture
def sample_code_file(valid_python_code, code_file_factory) -> CodeFile:
    """Return a sample CodeFile instance with valid Python code."""
    return code_file_factory(valid_python_code)


@pytest.fixture
def sample_pull_request(sample_code_file) -> PullRequest:
    """Return a sample PullRequest instance."""
    metadata = PullRequestMetadata(
        id="123",
        title="Add factorial and fibonacci functions",
        description="Implement utility math functions",
        author="test_user",
        base_branch="main",
        head_branch="feature/math-utils",
        created_at="2023-01-01T00:00:00Z",
        updated_at="2023-01-01T01:00:00Z",
    )

    pr = PullRequest(metadata=metadata)
    pr.add_file(sample_code_file)

    return pr


@pytest.fixture
def review_comment_factory():
    """Factory fixture to create ReviewComment instances."""

    def _create_comment(
        message: str,
        file_path: str = "test_file.py",
        line_start: int = 1,
        severity: SeverityLevel = SeverityLevel.MEDIUM,
        suggestion: str = None,
    ) -> ReviewComment:
        location = CodeLocation(file_path=file_path, line_start=line_start)

        return ReviewComment(
            severity=severity, location=location, message=message, suggestion=suggestion
        )

    return _create_comment


@pytest.fixture
def sample_review_comments(review_comment_factory) -> list[ReviewComment]:
    """Return a list of sample ReviewComment instances."""
    return [
        review_comment_factory(
            message="Missing type hints for function parameters and return values",
            line_start=2,
            severity=SeverityLevel.LOW,
            suggestion="Add type hints: def factorial(n: int) -> int:",
        ),
        review_comment_factory(
            message="Consider using iterative approach for factorial to avoid stack overflow",
            line_start=2,
            severity=SeverityLevel.MEDIUM,
            suggestion="Implement factorial iteratively using a loop",
        ),
        review_comment_factory(
            message="Good implementation of fibonacci using iteration",
            line_start=9,
            severity=SeverityLevel.INFO,
        ),
    ]
