"""
Tests for the base agent classes.
"""

import pytest
from unittest.mock import AsyncMock

from symposium.agents.base_agent import BaseAgent, System1Agent, SystemAgent
from symposium.models.base import (
    CodeFile,
    CodeLocation,
    FileMetadata,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
    SeverityLevel,
)


class TestBaseAgent:
    """Tests for the BaseAgent abstract class."""

    def test_creation(self):
        """Test creating a concrete BaseAgent implementation."""

        class ConcreteAgent(BaseAgent):
            async def process(self, input_data):
                return []

            async def generate_output(self):
                return {"result": "test"}

        agent = ConcreteAgent(name="Test")
        assert agent.name == "Test"
        assert str(agent) == "Test Agent"

    def test_abstract_methods(self):
        """Test that abstract methods are enforced."""
        with pytest.raises(TypeError):
            # Cannot instantiate abstract class
            BaseAgent(name="Test")


class TestSystem1Agent:
    """Tests for the System1Agent abstract class."""

    def test_process_with_pull_request(self, sample_pull_request):
        """Test processing a pull request."""

        class ConcreteSystem1Agent(System1Agent):
            async def analyze_file(self, file):
                # Simple mock implementation
                return [
                    ReviewComment(
                        severity=SeverityLevel.MEDIUM,
                        location=CodeLocation(
                            file_path=file.metadata.path, line_start=1
                        ),
                        message=f"Test comment for {file.metadata.path}",
                    )
                ]

            async def generate_output(self):
                return {"comments": []}

        agent = ConcreteSystem1Agent(name="TestSystem1")

        # Create a mock to avoid actual async execution
        agent.analyze_file = AsyncMock(
            return_value=[
                ReviewComment(
                    severity=SeverityLevel.MEDIUM,
                    location=CodeLocation(file_path="test_file.py", line_start=1),
                    message="Test comment",
                )
            ]
        )

        # Process the pull request
        import asyncio

        comments = asyncio.run(agent.process(sample_pull_request))

        # Check that analyze_file was called for each file
        assert agent.analyze_file.call_count == len(sample_pull_request.files)

        # Check that comments were tagged with the agent name
        assert len(comments) == len(sample_pull_request.files)
        for comment in comments:
            assert comment.source_agent == "TestSystem1"

    def test_process_with_file_list(self):
        """Test processing a list of files."""

        class ConcreteSystem1Agent(System1Agent):
            async def analyze_file(self, file):
                # Simple mock implementation
                return [
                    ReviewComment(
                        severity=SeverityLevel.MEDIUM,
                        location=CodeLocation(
                            file_path=file.metadata.path, line_start=1
                        ),
                        message=f"Test comment for {file.metadata.path}",
                    )
                ]

            async def generate_output(self):
                return {"comments": []}

        agent = ConcreteSystem1Agent(name="TestSystem1")

        # Create files
        files = [
            CodeFile(
                content="def test1():\n    return True",
                metadata=FileMetadata(path="test1.py", language="python"),
            ),
            CodeFile(
                content="def test2():\n    return False",
                metadata=FileMetadata(path="test2.py", language="python"),
            ),
        ]

        # Create a mock to avoid actual async execution
        agent.analyze_file = AsyncMock(
            side_effect=lambda file: [
                ReviewComment(
                    severity=SeverityLevel.MEDIUM,
                    location=CodeLocation(file_path=file.metadata.path, line_start=1),
                    message=f"Test comment for {file.metadata.path}",
                )
            ]
        )

        # Process the file list
        import asyncio

        comments = asyncio.run(agent.process(files))

        # Check that analyze_file was called for each file
        assert agent.analyze_file.call_count == len(files)

        # Check that comments were tagged with the agent name
        assert len(comments) == len(files)
        for i, comment in enumerate(comments):
            assert comment.source_agent == "TestSystem1"
            assert comment.message == f"Test comment for {files[i].metadata.path}"


class TestSystemAgent:
    """Tests for the SystemAgent abstract class."""

    def test_creation(self):
        """Test creating a concrete SystemAgent implementation."""

        class ConcreteSystemAgent(SystemAgent):
            async def process(self, input_data):
                return []

            async def generate_output(self):
                return {"result": "test"}

        # Valid system levels
        for level in range(2, 6):
            agent = ConcreteSystemAgent(name=f"System{level}", system_level=level)
            assert agent.name == f"System{level}"
            assert agent.system_level == level

    def test_invalid_system_level(self):
        """Test that invalid system levels are rejected."""

        class ConcreteSystemAgent(SystemAgent):
            async def process(self, input_data):
                return []

            async def generate_output(self):
                return {"result": "test"}

        # Invalid system levels
        with pytest.raises(ValueError):
            ConcreteSystemAgent(name="InvalidSystem", system_level=1)

        with pytest.raises(ValueError):
            ConcreteSystemAgent(name="InvalidSystem", system_level=6)
