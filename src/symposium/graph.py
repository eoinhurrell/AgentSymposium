from typing import Dict, List, Any, Optional, Tuple
import os
from pathlib import Path
import argparse
from datetime import datetime

from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, END

from symposium.models.base import (
    SeverityLevel,
    CodeFile,
    FileMetadata,
    PullRequest,
    PullRequestMetadata,
)
from symposium.agents import OperationalAgent, CoordinationAgent, SymposiumState


def create_symposium_workflow(
    llm,
    operational_agent_prompt: Optional[str] = None,
    coordination_agent_prompt: Optional[str] = None,
):
    """Create the Symposium LangGraph workflow with the specified agents."""

    # Initialize agents
    operational_agent = OperationalAgent(name="OperationalReviewer", llm=llm)

    coordination_agent = CoordinationAgent(name="CoordinationReviewer", llm=llm)

    # Create the workflow graph
    workflow = StateGraph(SymposiumState)

    # Add nodes
    workflow.add_node("operational_review", operational_agent.process)
    workflow.add_node("coordination_review", coordination_agent.process)

    # Define the workflow edges
    workflow.add_edge("operational_review", "coordination_review")
    workflow.add_edge("coordination_review", END)

    # Set the entry point
    workflow.set_entry_point("operational_review")

    return workflow.compile()


def load_code_files(
    directory_path: str,
    file_extensions: List[str] = [
        ".py",
        # ".js",
        # ".ts",
        # ".java",
        # ".go",
        # ".html",
        # ".css",
        ".md",
    ],
) -> List[Tuple[str, str]]:
    """
    Load code files from a directory with specified extensions.

    Args:
        directory_path: Path to the directory containing code files
        file_extensions: List of file extensions to include

    Returns:
        List of tuples containing (file_path, file_content)
    """
    code_files = []

    for root, _, files in os.walk(directory_path):
        for file in files:
            if any(file.endswith(ext) for ext in file_extensions):
                file_path = os.path.join(root, file)
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    try:
                        content = f.read()
                        relative_path = os.path.relpath(file_path, directory_path)
                        code_files.append((relative_path, content))
                    except Exception as e:
                        print(f"Error reading file {file_path}: {e}")

    return code_files


def detect_language(file_path: str) -> str:
    """Detect the programming language based on file extension."""
    ext = os.path.splitext(file_path)[1].lower()

    language_map = {
        ".py": "python",
        ".js": "javascript",
        ".ts": "typescript",
        ".jsx": "jsx",
        ".tsx": "tsx",
        ".html": "html",
        ".css": "css",
        ".java": "java",
        ".c": "c",
        ".cpp": "cpp",
        ".cs": "csharp",
        ".go": "go",
        ".rs": "rust",
        ".rb": "ruby",
        ".php": "php",
        ".md": "markdown",
    }

    return language_map.get(ext, "plaintext")


def create_pull_request(
    project_name: str, code_files: List[Tuple[str, str]]
) -> PullRequest:
    """
    Create a PullRequest object from a list of code files.

    Args:
        project_name: Name of the project
        code_files: List of tuples containing (file_path, file_content)

    Returns:
        PullRequest object
    """
    pr_metadata = PullRequestMetadata(
        id=f"PR-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        title=f"Code Review for {project_name}",
        description=f"Automated code review for {project_name} - {len(code_files)} files",
        author="Symposium",
        base_branch="main",
        head_branch="review",
        created_at=datetime.now().isoformat(),
        updated_at=datetime.now().isoformat(),
    )

    pull_request = PullRequest(metadata=pr_metadata)

    for file_path, content in code_files:
        language = detect_language(file_path)

        file_metadata = FileMetadata(
            path=file_path,
            language=language,
            is_new=False,
            line_count=len(content.splitlines()),
        )

        code_file = CodeFile(content=content, metadata=file_metadata)

        pull_request.add_file(code_file)

    return pull_request


def generate_markdown_report(state, output_file: str) -> None:
    """
    Generate a markdown report from the review comments.

    Args:
        state: The final state containing review comments
        output_file: Path to the output markdown file
    """
    pr_title = "Code Review"
    if state["pull_request"] and state["pull_request"].metadata.title:
        pr_title = state["pull_request"].metadata.title

    with open(output_file, "w", encoding="utf-8") as f:
        # Write header
        f.write(f"# {pr_title}\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # Executive summary (if available)
        executive_summaries = [
            comment
            for comment in state["pull_request"].comments
            if comment.location.file_path == "OVERALL_SUMMARY"
        ]

        if executive_summaries:
            f.write("## Executive Summary\n\n")
            f.write(executive_summaries[0].message)
            f.write("\n\n")

        # Group comments by file
        comments_by_file = {}
        for comment in state["pull_request"].comments:
            if comment.location.file_path == "OVERALL_SUMMARY":
                continue

            file_path = comment.location.file_path
            if file_path not in comments_by_file:
                comments_by_file[file_path] = []
            comments_by_file[file_path].append(comment)

        # File summaries first
        for file_path, comments in comments_by_file.items():
            summary_comments = [c for c in comments if c.category == "Summary"]
            if summary_comments:
                f.write(f"## {file_path} Summary\n\n")
                f.write(summary_comments[0].message)
                f.write("\n\n")

        # Detailed comments by file and severity
        f.write("## Detailed Comments\n\n")

        for file_path, comments in comments_by_file.items():
            f.write(f"### {file_path}\n\n")

            # Filter out summary comments
            detailed_comments = [c for c in comments if c.category != "Summary"]

            # Group by severity
            for severity in [
                SeverityLevel.CRITICAL,
                SeverityLevel.HIGH,
                SeverityLevel.MEDIUM,
                SeverityLevel.LOW,
                SeverityLevel.INFO,
            ]:
                severity_comments = [
                    c for c in detailed_comments if c.severity == severity
                ]

                if severity_comments:
                    f.write(f"#### {severity.value.upper()} Issues\n\n")

                    for i, comment in enumerate(severity_comments, 1):
                        f.write(
                            f"**Issue {i}** (Line {comment.location.line_start})\n\n"
                        )
                        f.write(f"{comment.message}\n\n")

                        if comment.suggestion:
                            f.write("**Suggestion:**\n\n")
                            f.write(f"{comment.suggestion}\n\n")

                        f.write("---\n\n")

        # Statistics
        f.write("## Review Statistics\n\n")

        # Count by severity
        severity_counts = {severity: 0 for severity in SeverityLevel}
        for comment in state["pull_request"].comments:
            if (
                comment.location.file_path != "OVERALL_SUMMARY"
                and comment.category != "Summary"
            ):
                severity_counts[comment.severity] += 1

        f.write("### Issues by Severity\n\n")
        for severity, count in severity_counts.items():
            if count > 0:
                f.write(f"- **{severity.value.upper()}**: {count} issues\n")

        f.write("\n")

        # Count by file
        f.write("### Issues by File\n\n")
        for file_path, comments in comments_by_file.items():
            detailed_count = len([c for c in comments if c.category != "Summary"])
            if detailed_count > 0:
                f.write(f"- **{file_path}**: {detailed_count} issues\n")


def main():
    """Main entry point for the Symposium workflow."""
    parser = argparse.ArgumentParser(description="Run Symposium code review workflow")
    parser.add_argument(
        "--directory",
        "-d",
        required=True,
        help="Directory containing code files to review",
    )
    parser.add_argument(
        "--output", "-o", default="review_report.md", help="Output markdown file"
    )
    parser.add_argument("--model", default="deepseek-r1:8b", help="Model to use")
    args = parser.parse_args()

    # Create LLM
    llm = ChatOllama(model=args.model, temperature=0)

    # Create workflow
    workflow = create_symposium_workflow(llm)

    # Load code files
    code_files = load_code_files(args.directory)
    print(f"Loaded {len(code_files)} code files from {args.directory}")

    # Create pull request
    project_name = os.path.basename(os.path.abspath(args.directory))
    pull_request = create_pull_request(project_name, code_files)

    # Create initial state
    initial_state = SymposiumState(pull_request=pull_request)

    # Run the workflow
    print("Running Symposium workflow...")
    final_state = workflow.invoke(initial_state)
    __import__("ipdb").set_trace()

    # Generate markdown report
    generate_markdown_report(final_state, args.output)
    print(f"Review report generated: {args.output}")


if __name__ == "__main__":
    main()
