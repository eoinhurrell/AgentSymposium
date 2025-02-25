"""
Helper functions for testing agents and their outputs.
"""

from typing import List, Optional, Set

from symposium.models.base import (
    CodeFile,
    ReviewComment,
    SeverityLevel,
)


def count_comments_by_severity(
    comments: List[ReviewComment], severity: Optional[SeverityLevel] = None
) -> int:
    """
    Count the number of comments with a specific severity.

    Args:
        comments: The list of comments to count.
        severity: The severity level to count. If None, count all comments.

    Returns:
        The number of comments with the specified severity.
    """
    if severity is None:
        return len(comments)

    return sum(1 for comment in comments if comment.severity == severity)


def find_comments_containing(
    comments: List[ReviewComment], text: str, case_sensitive: bool = False
) -> List[ReviewComment]:
    """
    Find comments containing specific text.

    Args:
        comments: The list of comments to search.
        text: The text to search for.
        case_sensitive: Whether to perform a case-sensitive search.

    Returns:
        A list of comments containing the specified text.
    """
    if not case_sensitive:
        text = text.lower()
        return [
            comment
            for comment in comments
            if text in comment.message.lower()
            or (comment.suggestion and text in comment.suggestion.lower())
        ]

    return [
        comment
        for comment in comments
        if text in comment.message
        or (comment.suggestion and text in comment.suggestion)
    ]


def find_comments_for_line(
    comments: List[ReviewComment], file_path: str, line: int
) -> List[ReviewComment]:
    """
    Find comments for a specific line in a file.

    Args:
        comments: The list of comments to search.
        file_path: The path of the file.
        line: The line number.

    Returns:
        A list of comments for the specified line.
    """
    return [
        comment
        for comment in comments
        if comment.location.file_path == file_path
        and comment.location.line_start <= line
        and (comment.location.line_end is None or comment.location.line_end >= line)
    ]


def has_duplicates(comments: List[ReviewComment]) -> bool:
    """
    Check if there are duplicate comments in the list.

    Args:
        comments: The list of comments to check.

    Returns:
        True if there are duplicates, False otherwise.
    """
    # Create a set of tuples containing the key fields of each comment
    unique_comments = set()

    for comment in comments:
        # Create a tuple of the key fields
        key = (comment.severity, str(comment.location), comment.message)

        if key in unique_comments:
            return True

        unique_comments.add(key)

    return False


def extract_line_from_file(file: CodeFile, line_number: int) -> Optional[str]:
    """
    Extract a specific line from a file.

    Args:
        file: The CodeFile to extract from.
        line_number: The line number to extract (1-indexed).

    Returns:
        The line content, or None if the line number is out of range.
    """
    try:
        return file.get_line(line_number)
    except IndexError:
        return None


def check_comment_context(comment: ReviewComment, file: CodeFile) -> bool:
    """
    Check if a comment's context matches the file content.

    Args:
        comment: The comment to check.
        file: The file the comment refers to.

    Returns:
        True if the context matches, False otherwise.
    """
    if not comment.context:
        return True

    try:
        line = file.get_line(comment.location.line_start)
        return line in comment.context
    except IndexError:
        return False
