"""
Python Syntax Agent for code review.

This agent analyzes Python code for syntax errors, style issues, and comment quality.
"""

import ast
import re
from typing import Dict, List, Optional, Tuple, Union

from symposium.agents.base_agent import System1Agent
from symposium.models.base import (
    CodeFile,
    CodeLocation,
    ReviewComment,
    SeverityLevel,
)


class PythonSyntaxAgent(System1Agent):
    """
    Agent that analyzes Python code for syntax errors and style issues.

    This agent checks for:
    1. Syntax errors using Python's ast module
    2. PEP 8 style issues (line length, indentation, naming conventions)
    3. Comment and docstring quality
    """

    def __init__(self, name: str = "PythonSyntax"):
        """Initialize the Python Syntax Agent."""
        super().__init__(name)
        self.max_line_length = 88  # PEP 8 recommends 79, but modern convention often uses 88 (black default)
        self.indentation_size = 4  # PEP 8 recommends 4 spaces

    async def analyze_file(self, file: CodeFile) -> List[ReviewComment]:
        """
        Analyze a Python file for syntax errors and style issues.

        Args:
            file: The Python file to analyze.

        Returns:
            List of ReviewComment objects with the analysis results.
        """
        if file.metadata.language.lower() != "python":
            return []

        comments = []

        # Run different types of analysis
        syntax_comments = await self._check_syntax(file)
        comments.extend(syntax_comments)

        # If there are syntax errors, don't bother with style checking
        if not any(
            comment.severity == SeverityLevel.CRITICAL for comment in syntax_comments
        ):
            style_comments = await self._check_style(file)
            comments.extend(style_comments)

            comment_comments = await self._check_comments(file)
            comments.extend(comment_comments)

        return comments

    async def _check_syntax(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for syntax errors in Python code.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for syntax errors.
        """
        comments = []

        try:
            # Try to parse the code
            ast.parse(file.content)
        except SyntaxError as e:
            # Create a comment for the syntax error
            location = CodeLocation(
                file_path=file.metadata.path, line_start=e.lineno, column_start=e.offset
            )

            message = f"Syntax error: {e.msg}"

            # Get the line with the error for context
            error_line = None
            try:
                error_line = file.get_line(e.lineno)
            except IndexError:
                pass

            context = error_line if error_line else None

            comment = ReviewComment(
                severity=SeverityLevel.CRITICAL,
                location=location,
                message=message,
                context=context,
                category="syntax",
            )

            comments.append(comment)

        return comments

    async def _check_style(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for PEP 8 style issues in Python code.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for style issues.
        """
        comments = []
        lines = file.get_lines()

        # Check line length
        line_length_comments = await self._check_line_length(file)
        comments.extend(line_length_comments)

        # Check indentation
        indentation_comments = await self._check_indentation(file)
        comments.extend(indentation_comments)

        # Check naming conventions
        naming_comments = await self._check_naming_conventions(file)
        comments.extend(naming_comments)

        return comments

    async def _check_line_length(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for lines exceeding the maximum length.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for line length issues.
        """
        comments = []
        lines = file.get_lines()

        for i, line in enumerate(lines, 1):
            # Skip empty lines or comment-only lines
            if not line.strip() or line.strip().startswith("#"):
                continue

            if len(line) > self.max_line_length:
                location = CodeLocation(file_path=file.metadata.path, line_start=i)

                message = (
                    f"Line too long ({len(line)} > {self.max_line_length} characters)"
                )
                suggestion = (
                    f"Consider breaking this line into multiple lines or "
                    f"refactoring to reduce length to {self.max_line_length} characters or fewer."
                )

                comment = ReviewComment(
                    severity=SeverityLevel.LOW,
                    location=location,
                    message=message,
                    suggestion=suggestion,
                    context=line,
                    category="style",
                )

                comments.append(comment)

        return comments

    async def _check_indentation(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for inconsistent indentation in the file.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for indentation issues.
        """
        comments = []
        lines = file.get_lines()

        # Detect if the file uses tabs or spaces
        indentation_type = "space"
        tab_pattern = re.compile(r"^\t+")
        space_pattern = re.compile(r"^ +")

        tab_lines = [i for i, line in enumerate(lines, 1) if tab_pattern.match(line)]
        space_lines = [
            i for i, line in enumerate(lines, 1) if space_pattern.match(line)
        ]

        # Check if mixed tabs and spaces are used
        if tab_lines and space_lines:
            # Find the first mixing instance
            first_tab_line = min(tab_lines) if tab_lines else float("inf")
            first_space_line = min(space_lines) if space_lines else float("inf")

            # Use the first line with indentation to determine the standard
            if first_tab_line < first_space_line:
                indentation_type = "tab"
                problematic_lines = space_lines
            else:
                indentation_type = "space"
                problematic_lines = tab_lines

            # Create a comment for the first instance of inconsistent indentation
            if problematic_lines:
                first_problematic_line = min(problematic_lines)
                location = CodeLocation(
                    file_path=file.metadata.path, line_start=first_problematic_line
                )

                message = (
                    f"Inconsistent indentation detected: This file mixes tabs and spaces. "
                    f"The file primarily uses {indentation_type}s for indentation."
                )
                suggestion = (
                    f"Use only {indentation_type}s for indentation throughout the file. "
                    f"PEP 8 recommends using 4 spaces for indentation."
                )

                context = (
                    lines[first_problematic_line - 1]
                    if first_problematic_line <= len(lines)
                    else None
                )

                comment = ReviewComment(
                    severity=SeverityLevel.MEDIUM,
                    location=location,
                    message=message,
                    suggestion=suggestion,
                    context=context,
                    category="style",
                )

                comments.append(comment)

        # Check indentation divisibility by standard size (if using spaces)
        if indentation_type == "space":
            for i, line in enumerate(lines, 1):
                if space_pattern.match(line):
                    leading_spaces = len(line) - len(line.lstrip(" "))
                    if leading_spaces % self.indentation_size != 0:
                        location = CodeLocation(
                            file_path=file.metadata.path, line_start=i
                        )

                        message = (
                            f"Non-standard indentation: {leading_spaces} spaces used. "
                            f"Indentation should be a multiple of {self.indentation_size} spaces."
                        )
                        suggestion = (
                            f"Use {self.indentation_size} spaces per indentation level."
                        )

                        comment = ReviewComment(
                            severity=SeverityLevel.LOW,
                            location=location,
                            message=message,
                            suggestion=suggestion,
                            context=line,
                            category="style",
                        )

                        comments.append(comment)

        return comments

    async def _check_naming_conventions(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for adherence to PEP 8 naming conventions.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for naming convention issues.
        """
        comments = []

        try:
            # Parse the file to get an AST
            tree = ast.parse(file.content)

            # Check class names (should be CamelCase)
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    if not re.match(r"^[A-Z][a-zA-Z0-9]*$", node.name):
                        location = CodeLocation(
                            file_path=file.metadata.path, line_start=node.lineno
                        )

                        message = (
                            f"Class name '{node.name}' does not follow PEP 8 naming convention. "
                            f"Class names should use CamelCase (CapWords) convention."
                        )
                        suggestion = f"Rename the class to follow CamelCase convention."

                        comment = ReviewComment(
                            severity=SeverityLevel.LOW,
                            location=location,
                            message=message,
                            suggestion=suggestion,
                            context=file.get_line(node.lineno),
                            category="style",
                        )

                        comments.append(comment)

                # Check function and method names (should be snake_case)
                elif isinstance(node, ast.FunctionDef):
                    # Skip special methods (e.g., __init__)
                    if not node.name.startswith("__") and not node.name.endswith("__"):
                        if not re.match(r"^[a-z][a-z0-9_]*$", node.name):
                            location = CodeLocation(
                                file_path=file.metadata.path, line_start=node.lineno
                            )

                            message = (
                                f"Function name '{node.name}' does not follow PEP 8 naming convention. "
                                f"Function names should use snake_case."
                            )
                            suggestion = (
                                f"Rename the function to follow snake_case convention."
                            )

                            comment = ReviewComment(
                                severity=SeverityLevel.LOW,
                                location=location,
                                message=message,
                                suggestion=suggestion,
                                context=file.get_line(node.lineno),
                                category="style",
                            )

                            comments.append(comment)

                # Check variable names (should be snake_case)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            # Skip constants (all uppercase with underscores)
                            if not re.match(r"^[A-Z][A-Z0-9_]*$", target.id):
                                # Check for snake_case
                                if not re.match(r"^[a-z][a-z0-9_]*$", target.id):
                                    location = CodeLocation(
                                        file_path=file.metadata.path,
                                        line_start=node.lineno,
                                    )

                                    message = (
                                        f"Variable name '{target.id}' does not follow PEP 8 naming convention. "
                                        f"Variable names should use snake_case."
                                    )
                                    suggestion = f"Rename the variable to follow snake_case convention."

                                    comment = ReviewComment(
                                        severity=SeverityLevel.LOW,
                                        location=location,
                                        message=message,
                                        suggestion=suggestion,
                                        context=file.get_line(node.lineno),
                                        category="style",
                                    )

                                    comments.append(comment)

        except SyntaxError:
            # If there's a syntax error, we've already reported it in _check_syntax
            pass

        return comments

    async def _check_comments(self, file: CodeFile) -> List[ReviewComment]:
        """
        Check for docstring and comment quality.

        Args:
            file: The Python file to check.

        Returns:
            List of ReviewComment objects for comment/docstring issues.
        """
        comments = []

        try:
            # Parse the file to get an AST
            tree = ast.parse(file.content)

            # Check module docstring
            if not ast.get_docstring(tree):
                location = CodeLocation(file_path=file.metadata.path, line_start=1)

                message = "Module is missing a docstring."
                suggestion = "Add a docstring at the beginning of the module to describe its purpose."

                comment = ReviewComment(
                    severity=SeverityLevel.MEDIUM,
                    location=location,
                    message=message,
                    suggestion=suggestion,
                    category="documentation",
                )

                comments.append(comment)

            # Check class and function docstrings
            for node in ast.walk(tree):
                if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                    if not ast.get_docstring(node):
                        node_type = (
                            "Class" if isinstance(node, ast.ClassDef) else "Function"
                        )
                        name = node.name

                        location = CodeLocation(
                            file_path=file.metadata.path, line_start=node.lineno
                        )

                        message = f"{node_type} '{name}' is missing a docstring."
                        suggestion = f"Add a docstring to describe the {node_type.lower()}'s purpose, parameters, and return value."

                        comment = ReviewComment(
                            severity=SeverityLevel.MEDIUM,
                            location=location,
                            message=message,
                            suggestion=suggestion,
                            context=file.get_line(node.lineno),
                            category="documentation",
                        )

                        comments.append(comment)

        except SyntaxError:
            # If there's a syntax error, we've already reported it in _check_syntax
            pass

        return comments
