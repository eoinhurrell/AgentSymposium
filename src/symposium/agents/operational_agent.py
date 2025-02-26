import re
import uuid
from typing import Any, Dict, List, Optional, Union, Set
from pydantic import BaseModel, Field

# LangChain imports
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain.tools import BaseTool
from langchain_ollama import ChatOllama

# LangGraph imports
from langgraph.graph import StateGraph, END

from symposium.models.base import (
    SeverityLevel,
    CodeFile,
    FileMetadata,
    CodeLocation,
    PullRequest,
    PullRequestMetadata,
    ReviewComment,
)


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


# ======= TOOLS FOR EXTERNAL SERVICES =======


class LintingTool(BaseTool):
    """Tool for linting code files."""

    name: str = "linting_tool"
    description: str = "Lints code files to find style issues and potential bugs."

    def _run(self, file_path: str, content: str, language: str) -> List[Dict[str, Any]]:
        """Run the linting tool on the given file."""
        # In a real implementation, this would call an external linting service
        # For demonstration, we'll return some simulated results

        issues = []

        # Simulate finding issues
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Check for long lines
            if len(line) > 100:
                issues.append(
                    {
                        "severity": "low",
                        "line": i + 1,
                        "message": "Line exceeds 100 characters.",
                        "suggestion": "Consider breaking this line into multiple lines.",
                    }
                )

            # Check for TODOs
            if "TODO" in line:
                issues.append(
                    {
                        "severity": "info",
                        "line": i + 1,
                        "message": "TODO found in code.",
                        "suggestion": "Consider addressing this TODO before submitting.",
                    }
                )

        return issues


class SecurityScannerTool(BaseTool):
    """Tool for scanning code for security vulnerabilities."""

    name: str = "security_scanner"
    description: str = "Scans code for security vulnerabilities."

    def _run(self, file_path: str, content: str, language: str) -> List[Dict[str, Any]]:
        """Run the security scanner on the given file."""
        # In a real implementation, this would call an external security scanner

        vulnerabilities = []

        # Simulate finding vulnerabilities
        lines = content.split("\n")
        for i, line in enumerate(lines):
            # Check for potential SQL injection
            if (
                "execute(" in line
                and "'" in line
                and language in ["python", "javascript"]
            ):
                vulnerabilities.append(
                    {
                        "severity": "high",
                        "line": i + 1,
                        "message": "Potential SQL injection vulnerability detected.",
                        "suggestion": "Use parameterized queries instead of string concatenation.",
                    }
                )

            # Check for hardcoded credentials
            if re.search(r"password|secret|key.*=.*['\"]", line, re.IGNORECASE):
                vulnerabilities.append(
                    {
                        "severity": "critical",
                        "line": i + 1,
                        "message": "Hardcoded credentials detected.",
                        "suggestion": "Move sensitive information to environment variables or a secure vault.",
                    }
                )

        return vulnerabilities


class ComplexityAnalyzerTool(BaseTool):
    """Tool for analyzing code complexity."""

    name: str = "complexity_analyzer"
    description: str = "Analyzes code complexity metrics."

    def _run(self, file_path: str, content: str, language: str) -> List[Dict[str, Any]]:
        """Run the complexity analyzer on the given file."""
        # In a real implementation, this would calculate complexity metrics

        complexity_issues = []

        lines = content.split("\n")

        # Simple function detection and analysis
        if language == "python":
            current_function = None
            function_start_line = 0

            for i, line in enumerate(lines):
                if re.match(r"^\s*def\s+\w+\s*\(", line):
                    if current_function and i - function_start_line > 30:
                        # Function is too long
                        complexity_issues.append(
                            {
                                "severity": "medium",
                                "start_line": function_start_line + 1,
                                "end_line": i,
                                "message": f"Function '{current_function}' is too long ({i - function_start_line} lines).",
                                "suggestion": "Consider breaking this function into smaller, more focused functions.",
                            }
                        )

                    # Extract function name
                    match = re.search(r"def\s+(\w+)", line)
                    if match:
                        current_function = match.group(1)
                        function_start_line = i

        # Detect deep nesting
        for i, line in enumerate(lines):
            indentation = len(line) - len(line.lstrip())
            if indentation > 12 and re.match(
                r"^\s*if\s+", line
            ):  # More than 3 levels of indentation
                complexity_issues.append(
                    {
                        "severity": "medium",
                        "start_line": i + 1,
                        "end_line": i + 1,
                        "message": "Deeply nested conditional detected.",
                        "suggestion": "Consider refactoring to reduce nesting, possibly using guard clauses or extracting methods.",
                    }
                )

        return complexity_issues


# ======= VSM SYSTEM 1: OPERATIONS =======
# Functions that perform specific review tasks


def system1_lint_code(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 1: Operation
    Lints the code files using an external linting tool via an LLM that can call it when needed.
    """
    print("LINT")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Get LLM from state context if available
    # llm = new_state.context.get("llm")
    llm = ChatOllama(model="deepseek-r1:8b")

    # Create the linting tool
    linting_tool = LintingTool()

    # If no LLM is available, fall back to simple linting
    if not llm:
        for file in new_state.pull_request.files:
            # Call the linting tool directly
            issues = linting_tool._run(
                file.metadata.path, file.content, file.metadata.language or ""
            )

            # Convert the issues to review comments
            for issue in issues:
                comment = ReviewComment(
                    id=str(uuid.uuid4()),
                    severity=SeverityLevel(issue["severity"]),
                    location=CodeLocation(
                        file_path=file.metadata.path, line_start=issue["line"]
                    ),
                    message=issue["message"],
                    suggestion=issue.get("suggestion"),
                    source_agent="linter",
                    category="style",
                )
                new_state.pull_request.comments.append(comment)

        new_state.messages.append(
            AIMessage(
                content=f"Performed basic linting and found {len(new_state.pull_request.comments)} issues."
            )
        )

        return new_state

    # Set up the LLM with linting tool integration
    try:
        import json
        from langchain.tools import Tool
        from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
        from langchain.agents import tool

        # Create a function that the LLM will use to call the linting tool
        @tool
        def lint_code(file_path: str, code_content: str, language: str) -> str:
            """
            Lints the provided code and finds style issues and potential bugs.

            Args:
                file_path: Path to the file being analyzed
                code_content: Content of the code file to analyze
                language: Programming language of the code (e.g., python, javascript)

            Returns:
                A list of linting issues with severity, line numbers, and suggestions
            """
            results = linting_tool._run(file_path, code_content, language)
            return json.dumps(results, indent=2)

        # Process each file with the LLM using the linting tool
        for file in new_state.pull_request.files:
            file_path = file.metadata.path
            content = file.content
            language = file.metadata.language or ""

            # Create a prompt for code style analysis that knows about the tool
            system_prompt = """You are an expert code reviewer focusing on code style and best practices.
            
            You have access to a linting tool you can call:
            lint_code(file_path: str, code_content: str, language: str) -> Returns linting issues

            Your process should be:
            1. Call the lint_code tool to get basic style issues
            2. Review the code yourself to find additional issues that the tool may have missed
            3. Provide a comprehensive style analysis including:
               - Naming conventions
               - Code structure and organization
               - Documentation and comments
               - Language-specific best practices
            
            Your response should be structured as JSON with this format:
            {
                "issues": [
                    {
                        "line": <line_number>,
                        "line_end": <optional_end_line_number>,
                        "message": "<description_of_the_issue>",
                        "severity": "<critical|high|medium|low|info>",
                        "suggestion": "<specific_improvement_recommendation>",
                        "category": "style",
                        "source": "<tool|expert>" // Was this found by the tool or your expert analysis
                    }
                ]
            }
            
            To call the linting tool, use this format:
            <tool>lint_code</tool>
            <tool_input>
            {
                "file_path": "path/to/file.py",
                "code_content": "def example():\\n    pass",
                "language": "python"
            }
            </tool_input>
            """

            # Create the user message with the code to analyze
            user_prompt = f"""
            Please analyze this {language} code file for style issues:
            
            File: {file_path}
            
            ```{language}
            {content}
            ```
            
            First use the linting tool, then enhance with your own expert analysis.
            Return a thorough review as structured JSON.
            """

            # Set up the conversation with the LLM
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]

            # Create a function to handle tool calls in the LLM's response
            def handle_tool_calls(response_text):
                import re

                # Look for tool calls in the response
                tool_pattern = r"<tool>(.*?)</tool>\s*<tool_input>(.*?)</tool_input>"
                tool_matches = re.finditer(tool_pattern, response_text, re.DOTALL)

                for match in tool_matches:
                    tool_name = match.group(1).strip()
                    tool_input_str = match.group(2).strip()

                    if tool_name == "lint_code":
                        try:
                            # Parse the tool input
                            tool_input = json.loads(tool_input_str)

                            # Call the lint_code function with the extracted parameters
                            tool_result = lint_code(
                                tool_input.get("file_path", file_path),
                                tool_input.get("code_content", content),
                                tool_input.get("language", language),
                            )

                            # Return the tool result
                            return tool_result
                        except Exception as e:
                            print(f"Error calling lint_code tool: {str(e)}")
                            return f"Error: {str(e)}"

                return None

            # Start the conversation with the LLM
            response = llm.invoke(messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # Check if the LLM wants to call the tool
            tool_result = handle_tool_calls(response_text)
            if tool_result:
                # Add the tool result to the conversation
                messages.append(AIMessage(content=f"Tool result: {tool_result}"))

                # Ask the LLM to continue with its analysis
                messages.append(
                    HumanMessage(
                        content="Now complete your analysis with both the tool results and your own expertise."
                    )
                )

                # Get the final response
                final_response = llm.invoke(messages)
                response_text = (
                    final_response.content
                    if hasattr(final_response, "content")
                    else str(final_response)
                )

            # Extract the structured JSON from the response
            try:
                # Look for JSON in the response
                import re

                json_match = re.search(
                    r"```(?:json)?\s*({.*?})\s*```", response_text, re.DOTALL
                )

                if json_match:
                    json_str = json_match.group(1)
                else:
                    # If not in code blocks, try to find a JSON object
                    json_match = re.search(r"({.*})", response_text, re.DOTALL)
                    json_str = json_match.group(1) if json_match else response_text

                # Parse the JSON
                analysis_result = json.loads(json_str)

                # Convert to ReviewComment objects
                for issue in analysis_result.get("issues", []):
                    # Map severity if needed
                    severity_str = issue.get("severity", "medium").lower()
                    if severity_str == "critical":
                        severity = SeverityLevel.CRITICAL
                    elif severity_str == "high":
                        severity = SeverityLevel.HIGH
                    elif severity_str == "medium":
                        severity = SeverityLevel.MEDIUM
                    elif severity_str == "low":
                        severity = SeverityLevel.LOW
                    else:
                        severity = SeverityLevel.INFO

                    # Create the comment
                    comment = ReviewComment(
                        id=str(uuid.uuid4()),
                        severity=severity,
                        location=CodeLocation(
                            file_path=file_path,
                            line_start=issue.get("line", 1),
                            line_end=issue.get("line_end", issue.get("line", 1)),
                        ),
                        message=issue["message"],
                        suggestion=issue.get("suggestion", "No suggestion provided"),
                        source_agent=f"linter_{'tool' if issue.get('source') == 'tool' else 'llm'}",
                        category=issue.get("category", "style"),
                    )
                    new_state.pull_request.comments.append(comment)

            except Exception as e:
                print(f"Error parsing LLM analysis: {str(e)}")
                # Add a fallback comment
                fallback_comment = ReviewComment(
                    id=str(uuid.uuid4()),
                    severity=SeverityLevel.LOW,
                    location=CodeLocation(file_path=file_path, line_start=1),
                    message=f"Error parsing style analysis for {file_path}",
                    suggestion="Please review this file manually for style issues.",
                    source_agent="linter_llm",
                    category="style",
                )
                new_state.pull_request.comments.append(fallback_comment)

    except Exception as e:
        print(f"Error in LLM-based linting: {str(e)}")
        # Fallback to basic linting if LLM integration fails
        for file in new_state.pull_request.files:
            issues = linting_tool._run(
                file.metadata.path, file.content, file.metadata.language or ""
            )

            for issue in issues:
                comment = ReviewComment(
                    id=str(uuid.uuid4()),
                    severity=SeverityLevel(issue["severity"]),
                    location=CodeLocation(
                        file_path=file.metadata.path, line_start=issue["line"]
                    ),
                    message=issue["message"],
                    suggestion=issue.get("suggestion"),
                    source_agent="linter",
                    category="style",
                )
                new_state.pull_request.comments.append(comment)

    # Add a message about the linting
    llm_status = "with LLM enhancement" if llm else "using basic tools"
    new_state.messages.append(
        AIMessage(
            content=f"Performed code style analysis {llm_status} and found {len(new_state.pull_request.comments)} issues."
        )
    )

    return new_state


def system1_security_check(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 1: Operation
    Checks for security vulnerabilities using an external security scanner.
    """
    print("SECURITY")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    security_scanner = SecurityScannerTool()
    initial_comments_count = len(new_state.pull_request.comments)

    for file in new_state.pull_request.files:
        # Call the security scanner
        vulnerabilities = security_scanner._run(
            file.metadata.path, file.content, file.metadata.language or ""
        )

        # Convert the vulnerabilities to review comments
        for vuln in vulnerabilities:
            comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel(vuln["severity"]),
                location=CodeLocation(
                    file_path=file.metadata.path, line_start=vuln["line"]
                ),
                message=vuln["message"],
                suggestion=vuln.get("suggestion"),
                source_agent="security_checker",
                category="security",
            )
            new_state.pull_request.comments.append(comment)

    # Add a message about the security check
    security_issues_count = (
        len(new_state.pull_request.comments) - initial_comments_count
    )
    new_state.messages.append(
        AIMessage(
            content=f"Performed security check and found {security_issues_count} security issues."
        )
    )

    return new_state


def system1_complexity_analysis(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 1: Operation
    Analyzes code complexity using an external analyzer tool.
    """
    print("COMPLEXITY")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    complexity_analyzer = ComplexityAnalyzerTool()
    initial_comments_count = len(new_state.pull_request.comments)

    for file in new_state.pull_request.files:
        # Call the complexity analyzer
        complexity_issues = complexity_analyzer._run(
            file.metadata.path, file.content, file.metadata.language or ""
        )

        # Convert the complexity issues to review comments
        for issue in complexity_issues:
            comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel(issue["severity"]),
                location=CodeLocation(
                    file_path=file.metadata.path,
                    line_start=issue["start_line"],
                    line_end=issue.get("end_line"),
                ),
                message=issue["message"],
                suggestion=issue.get("suggestion"),
                source_agent="complexity_analyzer",
                category="complexity",
            )
            new_state.pull_request.comments.append(comment)

    # Add a message about the complexity analysis
    complexity_issues_count = (
        len(new_state.pull_request.comments) - initial_comments_count
    )
    new_state.messages.append(
        AIMessage(
            content=f"Performed complexity analysis and found {complexity_issues_count} complexity issues."
        )
    )

    return new_state


# ======= VSM SYSTEM 2: COORDINATION =======
# Coordinates the operational units and resolves conflicts


def system2_coordinate_reviews(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 2: Coordination
    Coordinates the different reviews, removes duplicates, and groups related comments.
    """
    print("COORDINATION")
    new_state = state.model_copy(deep=True)

    # Group comments by location
    location_to_comments = {}
    for comment in new_state.pull_request.comments:
        location_str = str(comment.location)
        if location_str not in location_to_comments:
            location_to_comments[location_str] = []
        location_to_comments[location_str].append(comment)

    # Identify duplicates and related comments
    for location, comments in location_to_comments.items():
        if len(comments) > 1:
            # Sort by severity (highest first)
            sorted_by_severity = sorted(
                comments, key=lambda c: list(SeverityLevel).index(c.severity)
            )

            # Keep track of comment IDs to relate them
            comment_ids = [comment.id for comment in comments if comment.id]

            # Update related_comment_ids for all comments in this location
            for comment in comments:
                if comment.id:
                    related_ids = [cid for cid in comment_ids if cid != comment.id]
                    comment.related_comment_ids = related_ids

    # Add a message about the coordination
    locations_count = len(location_to_comments)
    new_state.messages.append(
        AIMessage(
            content=f"Coordinated {len(new_state.pull_request.comments)} review comments across {locations_count} locations."
        )
    )

    return new_state


# ======= VSM SYSTEM 3: CONTROL =======
# Manages the overall review process and resource allocation


def system3_control_review_process(
    state: CodeReviewState,
) -> Union[CodeReviewState, str]:
    """
    VSM System 3: Control
    Controls the review process, determines what to check next, and when to finish.
    """
    print("CONTROL")
    new_state = state.model_copy(deep=True)

    # Check if we have a pull request to review
    if not new_state.pull_request:
        new_state.messages.append(
            AIMessage(content="No pull request provided for review.")
        )
        return new_state

    # Check which review components have been run
    components_run = new_state.components_run

    # Determine the next component to run based on a predefined sequence
    if "linter" not in components_run:
        new_state.components_run = components_run.union({"linter"})
        new_state.current_agent = "linter"
        return new_state

    if "security_checker" not in components_run:
        new_state.components_run = components_run.union({"security_checker"})
        new_state.current_agent = "security_checker"
        return new_state

    if "complexity_analyzer" not in components_run:
        new_state.components_run = components_run.union({"complexity_analyzer"})
        new_state.current_agent = "complexity_analyzer"
        return new_state

    if "coordinator" not in components_run:
        new_state.components_run = components_run.union({"coordinator"})
        new_state.current_agent = "coordinator"
        return new_state

    if "intelligence" not in components_run:
        new_state.components_run = components_run.union({"intelligence"})
        new_state.current_agent = "intelligence"
        return new_state

    if "policy" not in components_run:
        new_state.components_run = components_run.union({"policy"})
        new_state.current_agent = "policy"
        return new_state

    # If all components have been run, we're done
    new_state.messages.append(AIMessage(content="Code review completed."))
    print("DONE")
    new_state.current_agent = ""
    return new_state


# ======= VSM SYSTEM 4: INTELLIGENCE =======
# Analyzes the broader context and environment


def system4_analyze_context(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 4: Intelligence
    Analyzes the broader context of the code changes, including PR metadata,
    repository trends, and development context.
    """
    print("ANALYZE")
    new_state = state.model_copy(deep=True)

    if not new_state.pull_request:
        return new_state

    # Analyze pull request metadata for context
    pr_metadata = new_state.pull_request.metadata

    # Check if the PR description is informative
    if not pr_metadata.description or len(pr_metadata.description) < 10:
        new_state.context["pr_description_quality"] = "poor"
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.LOW,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message="Pull request description is missing or too brief.",
            suggestion="Add a more detailed description explaining the purpose and context of these changes.",
            source_agent="intelligence",
            category="documentation",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["pr_description_quality"] = "good"

    # Analyze code changes as a whole
    total_lines_changed = sum(
        len(file.content.split("\n")) for file in new_state.pull_request.files
    )
    if total_lines_changed > 500:
        new_state.context["pr_size"] = "large"
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.MEDIUM,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message=f"This pull request is quite large ({total_lines_changed} lines).",
            suggestion="Consider breaking this into smaller, more focused pull requests for easier review.",
            source_agent="intelligence",
            category="process",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["pr_size"] = "reasonable"

    # Check if the PR affects critical components
    critical_paths = ["security", "auth", "payment", "core"]
    affected_critical_paths = []

    for file in new_state.pull_request.files:
        for path in critical_paths:
            if path in file.metadata.path.lower():
                affected_critical_paths.append(path)

    if affected_critical_paths:
        new_state.context["affects_critical_components"] = True
        paths_str = ", ".join(set(affected_critical_paths))
        comment = ReviewComment(
            id=str(uuid.uuid4()),
            severity=SeverityLevel.HIGH,
            location=CodeLocation(file_path="PULL_REQUEST", line_start=1),
            message=f"This PR affects critical components: {paths_str}",
            suggestion="Ensure thorough testing and consider requesting additional reviewers with expertise in these areas.",
            source_agent="intelligence",
            category="risk",
        )
        new_state.pull_request.comments.append(comment)
    else:
        new_state.context["affects_critical_components"] = False

    # Add a message about the context analysis
    files_count = len(new_state.pull_request.files)
    new_state.messages.append(
        AIMessage(
            content=f"Analyzed the context of the pull request with {files_count} files and {total_lines_changed} lines changed."
        )
    )

    return new_state


# ======= VSM SYSTEM 5: POLICY =======
# Makes high-level decisions and sets priorities


def system5_make_review_decision(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 5: Policy
    Makes overall decisions about the review, prioritizes comments, and provides a summary.
    """
    print("POLICY")
    new_state = state.model_copy(deep=True)

    if not new_state.pull_request:
        return new_state

    # Count issues by severity
    severity_counts = {
        SeverityLevel.CRITICAL: 0,
        SeverityLevel.HIGH: 0,
        SeverityLevel.MEDIUM: 0,
        SeverityLevel.LOW: 0,
        SeverityLevel.INFO: 0,
    }

    for comment in new_state.pull_request.comments:
        severity_counts[comment.severity] += 1

    # Determine overall recommendation based on policy
    if severity_counts[SeverityLevel.CRITICAL] > 0:
        recommendation = (
            "Changes required - Critical issues must be fixed before merging."
        )
    elif severity_counts[SeverityLevel.HIGH] > 0:
        recommendation = "Changes requested - High severity issues should be addressed."
    elif severity_counts[SeverityLevel.MEDIUM] > 3:
        recommendation = "Changes suggested - Several medium severity issues found."
    else:
        recommendation = (
            "Approved with comments - Minor issues can be addressed in follow-up PRs."
        )

    # Consider context from System 4
    if new_state.context.get("affects_critical_components", False):
        recommendation = "Changes required - Affects critical components and requires thorough review."

    # Create a summary comment
    summary = f"""
# Code Review Summary

## Overview
- Files reviewed: {len(new_state.pull_request.files)}
- Issues found: {sum(severity_counts.values())}
  - Critical: {severity_counts[SeverityLevel.CRITICAL]}
  - High: {severity_counts[SeverityLevel.HIGH]}
  - Medium: {severity_counts[SeverityLevel.MEDIUM]}
  - Low: {severity_counts[SeverityLevel.LOW]}
  - Informational: {severity_counts[SeverityLevel.INFO]}

## Recommendation
{recommendation}

## Key Issues to Address
"""

    # Add top issues to the summary
    critical_and_high = [
        c
        for c in new_state.pull_request.comments
        if c.severity in [SeverityLevel.CRITICAL, SeverityLevel.HIGH]
    ]
    for comment in critical_and_high[:5]:  # Top 5 critical/high issues
        summary += f"- {comment}\n"

    # If no critical/high issues, mention some medium ones
    if not critical_and_high and severity_counts[SeverityLevel.MEDIUM] > 0:
        medium_issues = [
            c
            for c in new_state.pull_request.comments
            if c.severity == SeverityLevel.MEDIUM
        ]
        for comment in medium_issues[:3]:  # Top 3 medium issues
            summary += f"- {comment}\n"

    # Store the summary in the state
    new_state.outputs["review_summary"] = summary
    new_state.outputs["recommendation"] = recommendation

    # Add summary message
    new_state.messages.append(AIMessage(content=summary))

    return new_state


# ======= LANGRAPH AGENT CREATION =======


def create_code_review_agent():
    """
    Creates a LangGraph agent for code review based on the Viable Systems Model.

    The agent follows the VSM structure:
    - System 1 (Operations): Linting, security checking, complexity analysis
    - System 2 (Coordination): Coordinating review comments
    - System 3 (Control): Managing the review process
    - System 4 (Intelligence): Analyzing PR context and environment
    - System 5 (Policy): Making overall decisions and recommendations

    Returns:
        A compiled LangGraph workflow for code review
    """
    # Define the state graph
    workflow = StateGraph(CodeReviewState)

    # Add nodes for each VSM system
    # System 1: Operations
    workflow.add_node("lint_code", system1_lint_code)
    workflow.add_node("security_check", system1_security_check)
    workflow.add_node("complexity_analysis", system1_complexity_analysis)

    # System 2: Coordination
    workflow.add_node("coordinate_reviews", system2_coordinate_reviews)

    # System 3: Control
    workflow.add_node("control_review_process", system3_control_review_process)

    # System 4: Intelligence
    workflow.add_node("analyze_context", system4_analyze_context)

    # System 5: Policy
    workflow.add_node("make_review_decision", system5_make_review_decision)

    # Define the edges
    # Start with control as the entry point
    workflow.set_entry_point("control_review_process")

    # Connect control to the appropriate next nodes based on its decision
    workflow.add_conditional_edges(
        "control_review_process",
        lambda state: state.current_agent if state.current_agent else END,
        {
            "linter": "lint_code",
            "security_checker": "security_check",
            "complexity_analyzer": "complexity_analysis",
            "coordinator": "coordinate_reviews",
            "intelligence": "analyze_context",
            "policy": "make_review_decision",
            END: END,  # Handle END case explicitly
        },
    )

    # All operations (System 1) go back to control
    workflow.add_edge("lint_code", "control_review_process")
    workflow.add_edge("security_check", "control_review_process")
    workflow.add_edge("complexity_analysis", "control_review_process")

    # Coordination (System 2) goes back to control
    workflow.add_edge("coordinate_reviews", "control_review_process")

    # Intelligence (System 4) goes back to control
    workflow.add_edge("analyze_context", "control_review_process")

    # Policy (System 5) goes back to control for final decision
    workflow.add_edge("make_review_decision", "control_review_process")
    workflow.add_edge("control_review_process", END)

    # Compile the graph
    code_review_app = workflow.compile()

    return code_review_app


# ======= USAGE EXAMPLE =======


def run_code_review(pull_request: PullRequest, llm: Optional[ChatOllama] = None):
    """
    Runs a code review on the given pull request.

    Args:
        pull_request: The pull request to review
        llm: Optional language model for AI-powered reviews

    Returns:
        The final state containing review results
    """
    # Create the agent
    code_review_agent = create_code_review_agent()

    # Create initial state
    initial_state = CodeReviewState(
        pull_request=pull_request,
        messages=[
            SystemMessage(content="Code review agent analyzing pull request."),
            HumanMessage(
                content=f"Please review PR #{pull_request.metadata.id}: {pull_request.metadata.title}"
            ),
        ],
        # context={"llm": llm},
    )

    # Run the agent
    final_state = code_review_agent.invoke(initial_state)

    # Return the results
    return final_state


# Example usage
if __name__ == "__main__":
    # Create a sample pull request
    sample_pr = PullRequest(
        metadata=PullRequestMetadata(
            id="123",
            title="Add user authentication feature",
            description="This PR implements basic user authentication using JWT tokens.",
            author="john.doe",
            base_branch="main",
            head_branch="feature/auth",
            created_at="2025-02-25T10:00:00Z",
            updated_at="2025-02-26T09:30:00Z",
        ),
        files=[
            CodeFile(
                content="""
import jwt
from flask import request, jsonify

def login():
    username = request.json.get('username')
    password = request.json.get('password')
    
    # TODO: Implement proper password hashing
    if username == 'admin' and password == 'password123':
        secret_key = 'my_super_secret_key'
        token = jwt.encode({'user': username}, secret_key, algorithm='HS256')
        return jsonify({'token': token})
    
    return jsonify({'error': 'Invalid credentials'}), 401
                """,
                metadata=FileMetadata(
                    path="auth/login.py",
                    language="python",
                ),
            )
        ],
    )

    llm = ChatOllama(model="deepseek-r1:8b")
    # Run the code review
    result = run_code_review(sample_pr, llm=llm)

    __import__("ipdb").set_trace()
    # Print the review summary
    print(result.outputs.get("review_summary", "No summary available"))
    pass
