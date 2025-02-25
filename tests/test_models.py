"""
Tests for the core data models.
"""

import pytest

from symposium.models.base import (
    CodeFile,
    CodeLocation,
    FileMetadata,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
    SeverityLevel,
)


class TestCodeLocation:
    """Tests for the CodeLocation model."""

    def test_creation(self):
        """Test creating a CodeLocation."""
        location = CodeLocation(file_path="test.py", line_start=10)
        assert location.file_path == "test.py"
        assert location.line_start == 10
        assert location.line_end is None
        assert location.column_start is None
        assert location.column_end is None

    def test_string_representation(self):
        """Test the string representation of a CodeLocation."""
        # Single line location
        location1 = CodeLocation(file_path="test.py", line_start=10)
        assert str(location1) == "test.py:10"

        # Multi-line location
        location2 = CodeLocation(file_path="test.py", line_start=10, line_end=15)
        assert str(location2) == "test.py:10-15"


class TestReviewComment:
    """Tests for the ReviewComment model."""

    def test_creation(self):
        """Test creating a ReviewComment."""
        location = CodeLocation(file_path="test.py", line_start=10)
        comment = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=location,
            message="This is a test comment",
            suggestion="Fix it this way",
        )

        assert comment.severity == SeverityLevel.HIGH
        assert comment.location.file_path == "test.py"
        assert comment.message == "This is a test comment"
        assert comment.suggestion == "Fix it this way"

    def test_default_values(self):
        """Test default values for ReviewComment."""
        location = CodeLocation(file_path="test.py", line_start=10)
        comment = ReviewComment(location=location, message="This is a test comment")

        assert comment.severity == SeverityLevel.MEDIUM
        assert comment.suggestion is None
        assert comment.source_agent is None
        assert comment.related_comment_ids == []

    def test_string_representation(self):
        """Test the string representation of a ReviewComment."""
        location = CodeLocation(file_path="test.py", line_start=10)

        # Comment without suggestion
        comment1 = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=location,
            message="This is a test comment",
        )
        assert str(comment1) == "[HIGH] test.py:10: This is a test comment"

        # Comment with suggestion
        comment2 = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=location,
            message="This is a test comment",
            suggestion="Fix it this way",
        )
        assert (
            str(comment2)
            == "[HIGH] test.py:10: This is a test comment\nSuggestion: Fix it this way"
        )


class TestFileMetadata:
    """Tests for the FileMetadata model."""

    def test_creation(self):
        """Test creating a FileMetadata."""
        metadata = FileMetadata(path="test.py", language="python")
        assert metadata.path == "test.py"
        assert metadata.language == "python"
        assert metadata.is_new is False
        assert metadata.is_deleted is False
        assert metadata.is_renamed is False

    def test_file_extension(self):
        """Test getting the file extension."""
        metadata1 = FileMetadata(path="test.py", language="python")
        assert metadata1.file_extension == ".py"

        metadata2 = FileMetadata(path="Dockerfile", language="dockerfile")
        assert metadata2.file_extension == ""

        metadata3 = FileMetadata(path="src/test.file.js", language="javascript")
        assert metadata3.file_extension == ".js"


class TestCodeFile:
    """Tests for the CodeFile model."""

    def test_creation(self):
        """Test creating a CodeFile."""
        metadata = FileMetadata(path="test.py", language="python")
        content = "def test():\n    return True"
        file = CodeFile(content=content, metadata=metadata)

        assert file.content == content
        assert file.metadata.path == "test.py"

    def test_get_lines(self):
        """Test getting lines from a CodeFile."""
        metadata = FileMetadata(path="test.py", language="python")
        content = "def test():\n    return True"
        file = CodeFile(content=content, metadata=metadata)

        lines = file.get_lines()
        assert len(lines) == 2
        assert lines[0] == "def test():"
        assert lines[1] == "    return True"

    def test_get_line(self):
        """Test getting a specific line from a CodeFile."""
        metadata = FileMetadata(path="test.py", language="python")
        content = "def test():\n    return True"
        file = CodeFile(content=content, metadata=metadata)

        assert file.get_line(1) == "def test():"
        assert file.get_line(2) == "    return True"

        with pytest.raises(IndexError):
            file.get_line(0)  # Line numbers are 1-indexed

        with pytest.raises(IndexError):
            file.get_line(3)  # Out of range


class TestPullRequest:
    """Tests for the PullRequest model."""

    def test_creation(self):
        """Test creating a PullRequest."""
        metadata = PullRequestMetadata(
            id="123",
            title="Test PR",
            author="testuser",
            base_branch="main",
            head_branch="feature",
            created_at="2023-01-01T00:00:00Z",
            updated_at="2023-01-01T01:00:00Z",
        )

        pr = PullRequest(metadata=metadata)
        assert pr.metadata.id == "123"
        assert pr.metadata.title == "Test PR"
        assert pr.files == []
        assert pr.comments == []

    def test_add_file(self):
        """Test adding a file to a PullRequest."""
        pr = PullRequest(
            metadata=PullRequestMetadata(
                id="123",
                title="Test PR",
                author="testuser",
                base_branch="main",
                head_branch="feature",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-01T01:00:00Z",
            )
        )

        file_metadata = FileMetadata(path="test.py", language="python")
        file = CodeFile(content="def test():\n    return True", metadata=file_metadata)

        pr.add_file(file)
        assert len(pr.files) == 1
        assert pr.files[0].metadata.path == "test.py"

    def test_add_comment(self):
        """Test adding a comment to a PullRequest."""
        pr = PullRequest(
            metadata=PullRequestMetadata(
                id="123",
                title="Test PR",
                author="testuser",
                base_branch="main",
                head_branch="feature",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-01T01:00:00Z",
            )
        )

        location = CodeLocation(file_path="test.py", line_start=1)
        comment = ReviewComment(
            severity=SeverityLevel.MEDIUM, location=location, message="Test comment"
        )

        pr.add_comment(comment)
        assert len(pr.comments) == 1
        assert pr.comments[0].message == "Test comment"

    def test_get_file_by_path(self):
        """Test getting a file by path from a PullRequest."""
        pr = PullRequest(
            metadata=PullRequestMetadata(
                id="123",
                title="Test PR",
                author="testuser",
                base_branch="main",
                head_branch="feature",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-01T01:00:00Z",
            )
        )

        # Add two files
        file1 = CodeFile(
            content="def test1():\n    return True",
            metadata=FileMetadata(path="test1.py", language="python"),
        )
        file2 = CodeFile(
            content="def test2():\n    return False",
            metadata=FileMetadata(path="test2.py", language="python"),
        )

        pr.add_file(file1)
        pr.add_file(file2)

        # Test finding files
        found_file = pr.get_file_by_path("test1.py")
        assert found_file is not None
        assert found_file.metadata.path == "test1.py"

        # Test file that doesn't exist
        not_found = pr.get_file_by_path("nonexistent.py")
        assert not_found is None

    def test_get_comments_by_file(self):
        """Test getting comments by file from a PullRequest."""
        pr = PullRequest(
            metadata=PullRequestMetadata(
                id="123",
                title="Test PR",
                author="testuser",
                base_branch="main",
                head_branch="feature",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-01T01:00:00Z",
            )
        )

        # Add comments for different files
        comment1 = ReviewComment(
            severity=SeverityLevel.MEDIUM,
            location=CodeLocation(file_path="test1.py", line_start=1),
            message="Comment on file 1",
        )
        comment2 = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=CodeLocation(file_path="test2.py", line_start=1),
            message="Comment on file 2",
        )
        comment3 = ReviewComment(
            severity=SeverityLevel.LOW,
            location=CodeLocation(file_path="test1.py", line_start=2),
            message="Another comment on file 1",
        )

        pr.add_comment(comment1)
        pr.add_comment(comment2)
        pr.add_comment(comment3)

        # Test filtering comments by file
        file1_comments = pr.get_comments_by_file("test1.py")
        assert len(file1_comments) == 2
        assert all(c.location.file_path == "test1.py" for c in file1_comments)

        file2_comments = pr.get_comments_by_file("test2.py")
        assert len(file2_comments) == 1
        assert file2_comments[0].message == "Comment on file 2"

        # Test file with no comments
        no_comments = pr.get_comments_by_file("nonexistent.py")
        assert len(no_comments) == 0

    def test_get_comments_by_severity(self):
        """Test getting comments by severity from a PullRequest."""
        pr = PullRequest(
            metadata=PullRequestMetadata(
                id="123",
                title="Test PR",
                author="testuser",
                base_branch="main",
                head_branch="feature",
                created_at="2023-01-01T00:00:00Z",
                updated_at="2023-01-01T01:00:00Z",
            )
        )

        # Add comments with different severities
        comment1 = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=CodeLocation(file_path="test.py", line_start=1),
            message="High severity comment",
        )
        comment2 = ReviewComment(
            severity=SeverityLevel.MEDIUM,
            location=CodeLocation(file_path="test.py", line_start=2),
            message="Medium severity comment",
        )
        comment3 = ReviewComment(
            severity=SeverityLevel.HIGH,
            location=CodeLocation(file_path="test.py", line_start=3),
            message="Another high severity comment",
        )

        pr.add_comment(comment1)
        pr.add_comment(comment2)
        pr.add_comment(comment3)

        # Test filtering comments by severity
        high_comments = pr.get_comments_by_severity(SeverityLevel.HIGH)
        assert len(high_comments) == 2
        assert all(c.severity == SeverityLevel.HIGH for c in high_comments)

        medium_comments = pr.get_comments_by_severity(SeverityLevel.MEDIUM)
        assert len(medium_comments) == 1
        assert medium_comments[0].message == "Medium severity comment"

        # Test severity with no comments
        low_comments = pr.get_comments_by_severity(SeverityLevel.LOW)
        assert len(low_comments) == 0
