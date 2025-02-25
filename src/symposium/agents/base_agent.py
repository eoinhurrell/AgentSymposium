"""
Base agent interface for the VSM-based code review system.
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Protocol, Union

from symposium.models.base import CodeFile, PullRequest, ReviewComment


class BaseAgent(ABC):
    """Abstract base class for all agents in the system."""

    def __init__(self, name: str):
        """Initialize the agent with a name."""
        self.name = name

    @abstractmethod
    async def process(
        self, input_data: Union[PullRequest, List[ReviewComment], Dict[str, Any]]
    ) -> Union[List[ReviewComment], Dict[str, Any]]:
        """
        Process the input data and return results.

        Args:
            input_data: The input data to process. Could be a PullRequest,
                        list of ReviewComment objects, or a dictionary of data.

        Returns:
            List of ReviewComment objects or a dictionary with processed data.
        """
        pass

    @abstractmethod
    async def generate_output(self) -> Dict[str, Any]:
        """
        Generate formatted output from the agent's processing results.

        Returns:
            A dictionary containing the formatted output from this agent.
        """
        pass

    def __str__(self) -> str:
        """String representation of the agent."""
        return f"{self.name} Agent"


class System1Agent(BaseAgent):
    """Base class for System 1 (Operational) agents that perform direct code analysis."""

    @abstractmethod
    async def analyze_file(self, file: CodeFile) -> List[ReviewComment]:
        """
        Analyze a single file and return review comments.

        Args:
            file: The CodeFile to analyze.

        Returns:
            List of ReviewComment objects.
        """
        pass

    async def process(
        self, input_data: Union[PullRequest, List[CodeFile]]
    ) -> List[ReviewComment]:
        """
        Process files in a pull request or a list of files.

        Args:
            input_data: Either a PullRequest or a list of CodeFile objects.

        Returns:
            List of ReviewComment objects.
        """
        all_comments = []

        if isinstance(input_data, PullRequest):
            files = input_data.files
        else:
            files = input_data

        for file in files:
            file_comments = await self.analyze_file(file)
            for comment in file_comments:
                # Tag comments with the agent that created them
                comment.source_agent = self.name
            all_comments.extend(file_comments)

        return all_comments

    async def generate_output(self) -> Dict[str, Any]:
        """
        Generate a dictionary with the agent's results.

        Returns:
            Dictionary with agent name and comments.
        """
        # This method would typically use stored state from process()
        # For simplicity, this implementation is a placeholder
        return {"agent": self.name, "comments": []}  # Would contain the actual comments


class SystemAgent(BaseAgent):
    """
    Base class for higher-level system agents (Systems 2-5)
    that process the output of other agents.
    """

    def __init__(self, name: str, system_level: int):
        """
        Initialize the higher-level system agent.

        Args:
            name: The name of the agent.
            system_level: The VSM system level (2-5).
        """
        super().__init__(name)
        if not 2 <= system_level <= 5:
            raise ValueError("System level must be between 2 and 5")
        self.system_level = system_level
