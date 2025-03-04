"""
Tests for the PythonSyntaxAgent.
"""

import pytest

from syposium.agents.python_syntax_agent import PythonSyntaxAgent
from syposium.models.base import CodeFile, FileMetadata, SeverityLevel


class TestPythonSyntaxAgent:
    """Tests for the PythonSyntaxAgent."""

    @pytest.fixture
    def agent(self):
        """Create a PythonSyntaxAgent for testing."""
        return PythonSyntaxAgent()

    @pytest.mark.asyncio
    async def test_valid_python_code(self, agent, valid_python_code, code_file_factory):
        """Test with valid Python code - should not have syntax errors."""
        # Create a CodeFile with valid Python code
        file = code_file_factory(valid_python_code)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # There should be no syntax errors, but there could be style comments
        assert not any(
            comment.severity == SeverityLevel.CRITICAL for comment in comments
        )

        # The valid code doesn't have a module docstring, so there should be at least one comment
        docstring_comments = [c for c in comments if c.category == "documentation"]
        assert len(docstring_comments) > 0
        assert any("module docstring" in c.message.lower() for c in docstring_comments)

    @pytest.mark.asyncio
    async def test_syntax_error(
        self, agent, python_code_with_syntax_error, code_file_factory
    ):
        """Test with Python code containing syntax errors."""
        # Create a CodeFile with Python code containing syntax errors
        file = code_file_factory(python_code_with_syntax_error)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # There should be at least one critical-severity comment for syntax error
        syntax_errors = [c for c in comments if c.severity == SeverityLevel.CRITICAL]
        assert len(syntax_errors) > 0

        # The comment should have the category "syntax"
        assert all(c.category == "syntax" for c in syntax_errors)

        # The syntax error location should point to the line with the error
        # In our fixture, the error is in the factorial function's missing parenthesis
        assert any(c.location.line_start == 5 for c in syntax_errors)

    @pytest.mark.asyncio
    async def test_style_issues(
        self, agent, python_code_with_style_issues, code_file_factory
    ):
        """Test with Python code containing style issues."""
        # Create a CodeFile with Python code containing style issues
        file = code_file_factory(python_code_with_style_issues)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # Filter for style comments
        style_comments = [c for c in comments if c.category == "style"]

        # There should be multiple style issues
        assert len(style_comments) > 0

        # Check for specific style issues we know are in the fixture
        style_issues = [c.message.lower() for c in style_comments]
        style_issues_str = " ".join(style_issues)

        # Our fixture has indentation issues
        assert (
            any("indentation" in issue for issue in style_issues)
            or "indentation" in style_issues_str
        )

    @pytest.mark.asyncio
    async def test_line_length_check(self, agent, code_file_factory):
        """Test line length checking."""
        # Create a file with a line that exceeds the maximum length
        long_line = (
            "x = " + "a" * 100
        )  # This will create a line longer than the 88 character limit
        code = f"""
def function_with_long_line():
    {long_line}
    return x
"""
        file = code_file_factory(code)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # Filter for line length issues
        line_length_comments = [
            c
            for c in comments
            if c.category == "style" and "line too long" in c.message.lower()
        ]

        # There should be at least one line length issue
        assert len(line_length_comments) > 0

        # The comment should point to the line with the long line
        assert any(c.location.line_start == 3 for c in line_length_comments)

    @pytest.mark.asyncio
    async def test_naming_conventions(self, agent, code_file_factory):
        """Test naming convention checking."""
        # Create a file with naming convention violations
        code = """
def badFunction():
    return True

class lowercase_class:
    def __init__(self):
        self.BadVariable = 123
"""
        file = code_file_factory(code)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # Filter for naming convention issues
        naming_comments = [
            c
            for c in comments
            if c.category == "style" and "naming convention" in c.message.lower()
        ]

        # There should be multiple naming convention issues
        assert len(naming_comments) >= 2

        # Check specific naming issues
        naming_messages = [c.message.lower() for c in naming_comments]
        assert any("function name 'badfunction'" in msg for msg in naming_messages)
        assert any("class name 'lowercase_class'" in msg for msg in naming_messages)

    @pytest.mark.asyncio
    async def test_docstring_checking(self, agent, code_file_factory):
        """Test docstring checking."""
        # Create a file with missing docstrings
        code = """
def function_without_docstring():
    x = 1
    return x

class ClassWithoutDocstring:
    def method_without_docstring(self):
        pass
"""
        file = code_file_factory(code)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # Filter for docstring issues
        docstring_comments = [c for c in comments if c.category == "documentation"]

        # There should be at least 3 docstring issues:
        # 1. Module docstring
        # 2. Function docstring
        # 3. Class docstring
        # 4. Method docstring
        assert len(docstring_comments) >= 3

        # Check specific docstring issues
        docstring_messages = [c.message.lower() for c in docstring_comments]
        assert any("module is missing a docstring" in msg for msg in docstring_messages)
        assert any(
            "function 'function_without_docstring' is missing a docstring" in msg
            for msg in docstring_messages
        )
        assert any(
            "class 'classwithoutdocstring' is missing a docstring" in msg
            for msg in docstring_messages
        )

    @pytest.mark.asyncio
    async def test_indentation_issues(self, agent, code_file_factory):
        """Test indentation checking."""
        # Create a file with indentation issues
        code = """
def function_with_bad_indentation():
    x = 1
   y = 2  # Wrong indentation (3 spaces instead of 4)
    return x + y

def function_with_mixed_indentation():
    x = 1
\ty = 2  # Tab instead of spaces
    return x + y
"""
        file = code_file_factory(code)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # Filter for indentation issues
        indentation_comments = [
            c
            for c in comments
            if c.category == "style"
            and ("indentation" in c.message.lower() or "space" in c.message.lower())
        ]

        # There should be indentation issues
        assert len(indentation_comments) > 0

    @pytest.mark.asyncio
    async def test_non_python_file(self, agent, code_file_factory):
        """Test that the agent ignores non-Python files."""
        # Create a non-Python file
        file = CodeFile(
            content="Not a Python file",
            metadata=FileMetadata(path="file.txt", language="text"),
        )

        # Analyze the file
        comments = await agent.analyze_file(file)

        # There should be no comments for a non-Python file
        assert len(comments) == 0

    @pytest.mark.asyncio
    async def test_complete_analysis(
        self, agent, python_code_with_quality_issues, code_file_factory
    ):
        """Test complete analysis with a file containing multiple issues."""
        # Create a CodeFile with Python code containing quality issues
        file = code_file_factory(python_code_with_quality_issues)

        # Analyze the file
        comments = await agent.analyze_file(file)

        # There should be multiple comments of different categories
        categories = set(c.category for c in comments)
        assert len(categories) >= 2  # At least style and documentation

        # There should be comments about:
        # - Missing docstrings
        # - Function naming (f and g are too short)
        # - Global variables
        comment_texts = " ".join(c.message.lower() for c in comments)
        assert "docstring" in comment_texts
        assert any(
            (
                "function name 'f'" in c.message.lower()
                or "variable name 'f'" in c.message.lower()
            )
            for c in comments
        )
