import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List, Dict, Optional, Any
from langchain_ollama import ChatOllama

from symposium.tools import run_linter
from symposium.models.base import (
    SeverityLevel,
    CodeLocation,
    ReviewComment,
    CodeReviewState,
    ReviewResult,
)

# from langgraph.prebuilt import create_react_agent
# from langchain_openai import ChatOpenAI
# from langchain_core.tools import tool
#
# # Below is one example of a tool definition
# @tool
# def get_stock_price(symbol: str) -> dict:
#     """Fetch the current stock price for a given symbol.
#     Args:
#         symbol (str): The stock ticker symbol (e.g., "AAPL" for Apple Inc.).
#     Returns:
#         dict: A dictionary containing the stock price or an error message.
#     """
#     base_url = "https://financialmodelingprep.com/api/v3/quote-short"
#     params = {"symbol": symbol, "apikey": os.getenv("FMP_API_KEY")}
#
#     response = requests.get(base_url, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         if data:
#             return {"price": data[0]["price"]}
#     return {"error": "Unable to fetch stock price."}
#
# # Below is one example of a simple react agent
# financial_data_agent = create_react_agent(
#     ChatOpenAI(model="gpt-4o-mini"),
#     tools=[get_stock_price, get_company_profile, ...],
#     state_modifier="You are a financial data agent responsible for retrieving financial data using the provided API tools ...",
# )

# Centralized configuration
MODEL = "MFDoom/deepseek-r1-tool-calling"

# Common severity level descriptions for consistent agent evaluation
SEVERITY_DEFINITIONS = """
CRITICAL Severity Issues:
- Security Vulnerabilities: SQL injection, hard-coded credentials, authentication bypasses
- Data Loss Risks: Code that could delete user data, race conditions corrupting records
- System Stability: Infinite loops, memory leaks, deadlocks, resource exhaustion

HIGH Severity Issues:
- Performance Problems: O(n²) algorithms, blocking operations, N+1 query problems
- Maintainability Blockers: Functions over 100 lines, deeply nested logic, duplicated business logic
- Reliability Concerns: Insufficient error handling, inconsistent state management

MEDIUM Severity Issues:
- Code Quality Problems: Mixed responsibilities, inconsistent error handling, magic numbers
- Potential Bugs: Off-by-one errors, null references, incorrect order of operations
- Inefficient Implementations: Redundant computations, inefficient string operations

LOW Severity Issues:
- Minor Code Smells: Slightly long methods, inconsistent naming, commented-out code
- Style Inconsistencies: Mixed indentation, inconsistent bracing styles
- Small Optimization Opportunities: Inefficient data structures for small collections

INFO Severity Issues:
- Suggestions for Improvement: Alternative approaches, design patterns, language features
- Documentation Notes: Missing docs, outdated comments, unclear parameter names
- Best Practices Reminders: Additional tests, better naming, coding standards
"""


# Shared helper functions
def create_review_comment(
    file_path: str,
    line_start: int,
    message: str,
    suggestion: str,
    source_agent: str,
    category: str,
    severity: SeverityLevel,
    line_end: Optional[int] = None,
) -> ReviewComment:
    """Create a standardized review comment with a unique ID."""
    return ReviewComment(
        id=str(uuid.uuid4()),
        severity=severity,
        location=CodeLocation(
            file_path=file_path, line_start=line_start, line_end=line_end
        ),
        message=message,
        suggestion=suggestion,
        source_agent=source_agent,
        category=category,
    )


def add_error_comment(
    state: CodeReviewState, file_path: str, error: str, agent_name: str, category: str
):
    """Add an error comment when agent analysis fails."""
    error_comment = create_review_comment(
        file_path=file_path,
        line_start=1,
        message=f"Error during {agent_name} analysis: {error}",
        suggestion=f"Please review this file manually for {category} issues.",
        source_agent=f"{agent_name.lower().replace(' ', '_')}_error",
        category=category,
        severity=SeverityLevel.LOW,
    )
    state.pull_request.comments.append(error_comment)


def add_summary_message(
    state: CodeReviewState, agent_name: str, source_agent: str, issue_description: str
):
    """Add a standardized summary message after agent completion."""
    agent_comments = [
        c for c in state.pull_request.comments if c.source_agent == source_agent
    ]
    state.messages.append(
        AIMessage(
            content=f"The {agent_name} completed analysis and identified {len(agent_comments)} {issue_description}."
        )
    )


# ======= VSM SYSTEM 1: OPERATIONS =======
# Functions that perform specific review tasks


def code_architect_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Code Architect: Focuses on code structure, readability, maintainability, and DRY principles.
    Analyzes code organization, naming conventions, adherence to software design principles,
    and code style through integrated linting analysis.
    """
    print("CODE ARCHITECT ANALYZING")

    # Skip if there's no pull request to analyze
    if not state.pull_request:
        return state

    # Initialize LLM with structured output capability
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        ReviewResult, method="json_schema"
    )

    # Process each file in the pull request
    for file in state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # First run a linter to get automatic style and formatting insights
        try:
            linter_results = run_linter.func(code=content, language=language)
        except Exception as e:
            print(f"Error running linter: {str(e)}")
            linter_results = f"Error running linter: {str(e)}"

        # Calculate additional code metrics
        code_metrics = analyze_code_metrics(content, language)

        # Create a specialized prompt for architecture analysis
        system_prompt = f"""You are The Code Architect, an expert in software design principles with decades of experience building maintainable systems. Your primary focus is on code structure, readability, and long-term maintainability.

When reviewing code, carefully analyze:
1. How well the code is broken down into appropriate-sized, single-responsibility functions and classes
2. The clarity and descriptiveness of all variable, function, and class names
3. Whether the code follows DRY principles or contains unnecessary duplication
4. If the code strikes the right balance between reusability and unnecessary abstraction
5. Code style and formatting issues that impact readability
6. Proper indentation, bracket placement, and overall visual structure
7. Cyclomatic complexity and nesting depth of functions
8. Consistency in coding style and conventions throughout the file

You've been provided with:
1. Automated linter output highlighting potential style and syntax issues
2. Code metrics including function lengths, complexity scores, and nesting depths

For each issue, create a ReviewComment with the following structure:
- id: (Leave empty, it will be filled automatically)
- severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO based on the impact to code quality
- location: An object with:
  - file_path: The path to the file being analyzed
  - line_start: The line number where the issue starts
  - line_end: (Optional) The line number where the issue ends
- message: A clear description of the architecture/design/style issue that explains WHY this is a problem
- suggestion: A specific recommendation to fix the issue with a code example when appropriate
- source_agent: "code_architect"
- category: One of "structure", "naming", "duplication", "abstraction", "readability", "style", "complexity"

Be practical and actionable with your feedback. Focus on issues that have real impact on maintainability.
Don't just identify problems - explain why they matter and provide clear solutions.

For style issues, use these prefixes to indicate importance:
- No prefix: Important issues that should be fixed
- "~" prefix: Minor issues that could be ignored
- "~~" prefix: Very minor issues that will likely be ignored

{SEVERITY_DEFINITIONS}
"""

        # Create user message with the code to analyze
        user_prompt = f"""
Analyze this {language} code file for architecture, design, and style issues:

File: {file_path}

```{language}
{content}
```

The automated linter produced the following output:
```
{linter_results}
```

The code metrics analysis produced the following:
```
{code_metrics}
```

Return a list of ReviewComment objects for all significant issues found, focusing on:
- Function/class size and responsibilities
- Variable/function naming clarity
- Code duplication
- Appropriate abstraction level
- Overall readability and maintainability
- Style and formatting issues
- Control flow complexity

Only report issues that meaningfully impact code quality. Be selective with LOW severity issues.
Group similar style issues rather than listing each occurrence separately.
"""

        # Set up the conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Analyze the code with the LLM
            result = llm_with_structured_output.invoke(messages)

            # Add review comments to the state
            for comment in result.comments:
                # Set the file path if not already set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Assign a unique ID
                comment.id = str(uuid.uuid4())

                # Add to the pull request comments
                state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Code Architect review: {str(e)}")
            add_error_comment(state, file_path, str(e), "Code Architect", "structure")

    # Add a summary message
    add_summary_message(
        state,
        "Code Architect",
        "code_architect",
        "architecture, design, and style issues",
    )

    return state


def analyze_code_metrics(content: str, language: str) -> str:
    """
    Analyzes code to extract metrics like function lengths, complexity, and nesting depth.
    Returns a formatted string with the metrics.
    """
    # Split the code into lines for analysis
    lines = content.splitlines()
    total_lines = len(lines)

    # Basic metrics to start with
    metrics = {
        "total_lines": total_lines,
        "empty_lines": sum(1 for line in lines if not line.strip()),
        "comment_lines": sum(
            1
            for line in lines
            if line.strip().startswith(("#", "//", "/*", "*", "*/"))
            if line.strip()
        ),
    }

    # Calculate code to comment ratio
    if metrics["comment_lines"] > 0:
        metrics["code_to_comment_ratio"] = round(
            (total_lines - metrics["empty_lines"] - metrics["comment_lines"])
            / metrics["comment_lines"],
            2,
        )
    else:
        metrics["code_to_comment_ratio"] = "∞ (no comments)"

    # Attempt to identify functions/methods and their lengths
    function_info = extract_functions(content, language)

    # Estimate nesting depth
    max_nesting = estimate_max_nesting(lines, language)
    avg_line_length = (
        sum(len(line) for line in lines) / total_lines if total_lines > 0 else 0
    )

    # Format the metrics
    result = []
    result.append(f"File Statistics:")
    result.append(f"- Total lines: {metrics['total_lines']}")
    result.append(
        f"- Empty lines: {metrics['empty_lines']} ({round(metrics['empty_lines']/total_lines*100, 1)}% of total)"
    )
    result.append(
        f"- Comment lines: {metrics['comment_lines']} ({round(metrics['comment_lines']/total_lines*100, 1)}% of total)"
    )
    result.append(f"- Code to comment ratio: {metrics['code_to_comment_ratio']}")
    result.append(f"- Average line length: {round(avg_line_length, 1)} characters")
    result.append(f"- Maximum nesting depth: {max_nesting}")

    if function_info:
        result.append("\nFunction Statistics:")
        for func in function_info:
            result.append(
                f"- {func['name']}: {func['lines']} lines (lines {func['start']}-{func['end']})"
            )

        # Calculate average and max function length
        func_lengths = [func["lines"] for func in function_info]
        if func_lengths:
            avg_length = sum(func_lengths) / len(func_lengths)
            max_length = max(func_lengths)
            result.append(f"\nFunction Summary:")
            result.append(f"- Total functions: {len(function_info)}")
            result.append(f"- Average function length: {round(avg_length, 1)} lines")
            result.append(f"- Maximum function length: {max_length} lines")

            # Flag particularly long functions
            long_funcs = [func for func in function_info if func["lines"] > 30]
            if long_funcs:
                result.append("\nPotentially problematic functions:")
                for func in long_funcs:
                    result.append(
                        f"- {func['name']}: {func['lines']} lines (may be too long)"
                    )

    return "\n".join(result)


def extract_functions(content: str, language: str) -> List[Dict]:
    """
    Extracts function information from code.
    Returns a list of dictionaries with function names, start and end lines, and total lines.
    """
    # This is a simplified implementation that won't work for all languages
    # In a real system, you'd want to use a proper parser for each language

    functions = []

    # Very simple detection based on common patterns
    # Only works for basic function definitions in common languages
    lines = content.splitlines()

    current_func = None
    brace_count = 0

    for i, line in enumerate(lines):
        line_stripped = line.strip()

        # Simple function detection - this would need to be language-specific in a real system
        if current_func is None:
            if language.lower() in ["python"]:
                if line_stripped.startswith("def ") and ":" in line_stripped:
                    func_name = line_stripped[4:].split("(")[0].strip()
                    current_func = {"name": func_name, "start": i + 1, "lines": 1}
            elif language.lower() in [
                "javascript",
                "typescript",
                "java",
                "c",
                "cpp",
                "csharp",
                "go",
            ]:
                # This is very simplified and won't catch all function definitions
                if (
                    "function " in line_stripped
                    or "func " in line_stripped
                    or ") {" in line_stripped
                ):
                    # Try to extract the function name
                    if "function " in line_stripped:
                        func_name = (
                            line_stripped.split("function ")[1].split("(")[0].strip()
                        )
                    elif "func " in line_stripped:
                        func_name = (
                            line_stripped.split("func ")[1].split("(")[0].strip()
                        )
                    else:
                        # Try to get the part before the parenthesis
                        parts = line_stripped.split("(")[0].strip().split()
                        func_name = parts[-1] if parts else "unknown"

                    current_func = {"name": func_name, "start": i + 1, "lines": 1}
                    if "{" in line:
                        brace_count += line.count("{") - line.count("}")
        else:
            current_func["lines"] += 1

            # For brace languages, track opening and closing braces
            if language.lower() in [
                "javascript",
                "typescript",
                "java",
                "c",
                "cpp",
                "csharp",
                "go",
            ]:
                brace_count += line.count("{") - line.count("}")
                if brace_count == 0 and "}" in line:
                    current_func["end"] = i + 1
                    functions.append(current_func)
                    current_func = None

            # For Python, look for lines that are not indented (end of function)
            elif language.lower() in ["python"]:
                if (
                    i < len(lines) - 1
                    and not lines[i + 1].startswith(" ")
                    and not lines[i + 1].startswith("\t")
                    and lines[i + 1].strip()
                    and not lines[i + 1].strip().startswith("#")
                ):
                    current_func["end"] = i + 1
                    functions.append(current_func)
                    current_func = None

    # Handle the case where the last function doesn't get properly closed
    if current_func is not None:
        current_func["end"] = len(lines)
        functions.append(current_func)

    return functions


def estimate_max_nesting(lines: List[str], language: str) -> int:
    """
    Estimates the maximum nesting depth in the code.
    This is a simplified implementation that works by tracking indentation or braces.
    """
    max_depth = 0
    current_depth = 0

    if language.lower() in ["python"]:
        # For Python, track indentation level
        indent_size = 4  # Assuming 4 spaces per indentation level

        for line in lines:
            if line.strip() and not line.strip().startswith("#"):
                # Count leading spaces
                leading_spaces = len(line) - len(line.lstrip())
                # Calculate indentation level
                level = leading_spaces // indent_size
                max_depth = max(max_depth, level)

    elif language.lower() in [
        "javascript",
        "typescript",
        "java",
        "c",
        "cpp",
        "csharp",
        "go",
    ]:
        # For brace languages, track brace depth
        for line in lines:
            # Ignore comments
            if line.strip().startswith("//") or line.strip().startswith("/*"):
                continue

            # Count braces
            for char in line:
                if char == "{":
                    current_depth += 1
                    max_depth = max(max_depth, current_depth)
                elif char == "}":
                    current_depth = max(0, current_depth - 1)  # Avoid negative depth

    return max_depth


def performance_guardian_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Performance Guardian: Focuses on speed, performance, reliability, and scalability.
    Identifies potential bottlenecks, error handling issues, and scalability concerns.
    """
    print("PERFORMANCE GUARDIAN ANALYZING")

    # Skip if there's no pull request to analyze
    if not state.pull_request:
        return state

    # Initialize LLM with structured output capability
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        ReviewResult, method="json_schema"
    )

    # Process each file in the pull request
    for file in state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a specialized prompt for performance analysis
        system_prompt = f"""You are The Performance Guardian, a systems expert obsessed with efficiency, reliability, and scalability. Your role is to identify performance bottlenecks, reliability issues, and scalability concerns.

When analyzing code, focus on:
1. Algorithm efficiency and computational complexity (especially O(n²) or worse)
2. Resource usage efficiency (memory, CPU, network, database)
3. Error handling and resilience against failures
4. Scalability under heavy load or with large datasets
5. Concurrency issues, race conditions, and deadlocks

For each issue, create a ReviewComment with the following structure:
- id: (Leave empty, it will be filled automatically)
- severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO based on the performance impact
- location: An object with:
  - file_path: The path to the file being analyzed
  - line_start: The line number where the issue starts
  - line_end: (Optional) The line number where the issue ends
- message: A clear description of the performance/reliability issue with context on its impact
- suggestion: A specific recommendation with example code to fix the issue
- source_agent: "performance_guardian"
- category: One of "performance", "error_handling", "scalability", "resource_usage", "reliability"

Be precise about the performance impact. Estimate the difference when possible (e.g., "This could be 10x faster by...").
Prioritize issues that would have the greatest real-world impact under load or at scale.

{SEVERITY_DEFINITIONS}
"""

        # Create user message with the code to analyze
        user_prompt = f"""
Analyze this {language} code file for performance and reliability issues:

File: {file_path}

```{language}
{content}
```

Return a list of ReviewComment objects for all significant issues found, focusing on:
- Algorithmic inefficiencies
- Resource usage problems
- Error handling gaps
- Scalability limitations
- Reliability concerns

Only report issues that would meaningfully impact performance or reliability in production. 
Be specific about the impact in each comment.
"""

        # Set up the conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Analyze the code with the LLM
            result = llm_with_structured_output.invoke(messages)

            # Add review comments to the state
            for comment in result.comments:
                # Set the file path if not already set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Assign a unique ID
                comment.id = str(uuid.uuid4())

                # Add to the pull request comments
                state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Performance Guardian review: {str(e)}")
            add_error_comment(
                state, file_path, str(e), "Performance Guardian", "performance"
            )

    # Add a summary message
    add_summary_message(
        state,
        "Performance Guardian",
        "performance_guardian",
        "performance and reliability issues",
    )

    return state


def security_sentinel_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Security Sentinel: Focuses on security vulnerabilities and test coverage.
    Identifies potential attack vectors and ensures adequate test coverage for critical functionality.
    """
    print("SECURITY SENTINEL ANALYZING")

    # Skip if there's no pull request to analyze
    if not state.pull_request:
        return state

    # Initialize LLM with structured output capability
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        ReviewResult, method="json_schema"
    )

    # Process each file in the pull request
    for file in state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a specialized prompt for security analysis
        system_prompt = f"""You are The Security Sentinel, a cybersecurity expert with a background in penetration testing and secure coding practices. Your mission is to identify security vulnerabilities and testing gaps.

When reviewing code, thoroughly investigate:
1. Injection vulnerabilities (SQL, command, etc.)
2. Authentication and authorization weaknesses
3. Data exposure risks and improper handling of sensitive information
4. Input validation issues and sanitization gaps
5. Insufficient test coverage, especially around security-critical code

For each issue, create a ReviewComment with the following structure:
- id: (Leave empty, it will be filled automatically)
- severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO based on the security risk
- location: An object with:
  - file_path: The path to the file being analyzed
  - line_start: The line number where the issue starts
  - line_end: (Optional) The line number where the issue ends
- message: A clear description of the security vulnerability or testing gap
- suggestion: A specific recommendation with secure code example to fix the issue
- source_agent: "security_sentinel"
- category: One of "security", "authentication", "data_exposure", "test_coverage", "input_validation"

Think like an attacker. Consider how the code could be exploited and explain the attack vector.
For each vulnerability, indicate the potential impact if exploited.

{SEVERITY_DEFINITIONS}
"""

        # Create user message with the code to analyze
        user_prompt = f"""
Analyze this {language} code file for security vulnerabilities and testing gaps:

File: {file_path}

```{language}
{content}
```

Return a list of ReviewComment objects for all significant issues found, focusing on:
- Security vulnerabilities and attack vectors
- Authentication and authorization flaws
- Data exposure and privacy risks
- Input validation and sanitization issues
- Test coverage gaps, especially for security-critical code

Prioritize issues by security impact. Be specific about how each vulnerability could be exploited.
"""

        # Set up the conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Analyze the code with the LLM
            result = llm_with_structured_output.invoke(messages)

            # Add review comments to the state
            for comment in result.comments:
                # Set the file path if not already set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Assign a unique ID
                comment.id = str(uuid.uuid4())

                # Add to the pull request comments
                state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Security Sentinel review: {str(e)}")
            add_error_comment(state, file_path, str(e), "Security Sentinel", "security")

    # Add a summary message
    add_summary_message(
        state,
        "Security Sentinel",
        "security_sentinel",
        "security vulnerabilities and testing gaps",
    )

    return state


def integration_specialist_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Integration Specialist: Focuses on documentation, appropriate use of language features,
    avoiding reinventing the wheel, and architectural coherence.
    """
    print("INTEGRATION SPECIALIST ANALYZING")

    # Skip if there's no pull request to analyze
    if not state.pull_request:
        return state

    # Initialize LLM with structured output capability
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        ReviewResult, method="json_schema"
    )

    # Process each file in the pull request
    for file in state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a specialized prompt for integration analysis
        system_prompt = f"""You are The Integration Specialist, an expert on language best practices, documentation, and system integration. Your focus is ensuring code fits well within the larger ecosystem.

When reviewing code, focus on:
1. Documentation quality and completeness (docstrings, comments, explanations)
2. Appropriate use of language-specific features and idioms
3. Whether standard libraries or frameworks should be used instead of custom implementations
4. How well the code integrates with the overall project architecture

For each issue, create a ReviewComment with the following structure:
- id: (Leave empty, it will be filled automatically)
- severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO based on the integration impact
- location: An object with:
  - file_path: The path to the file being analyzed
  - line_start: The line number where the issue starts
  - line_end: (Optional) The line number where the issue ends
- message: A clear description of the documentation or integration issue
- suggestion: A specific recommendation with example to address the issue
- source_agent: "integration_specialist"
- category: One of "documentation", "reinventing_wheel", "language_usage", "architectural_alignment", "best_practices"

Be educational in your feedback. When suggesting language features or libraries, explain why they're better than the current implementation.
For documentation issues, provide examples of what good documentation would look like.

{SEVERITY_DEFINITIONS}
"""

        # Create user message with the code to analyze
        user_prompt = f"""
Analyze this {language} code file for integration and documentation issues:

File: {file_path}

```{language}
{content}
```

Return a list of ReviewComment objects for all significant issues found, focusing on:
- Documentation quality and completeness
- Usage of language features and idioms
- Reinventing functionality available in standard libraries
- Integration with project architecture
- Adherence to language-specific best practices

Be specific about how each issue affects maintainability, onboarding new developers, or system integration.
"""

        # Set up the conversation
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Analyze the code with the LLM
            result = llm_with_structured_output.invoke(messages)

            # Add review comments to the state
            for comment in result.comments:
                # Set the file path if not already set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Assign a unique ID
                comment.id = str(uuid.uuid4())

                # Add to the pull request comments
                state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Integration Specialist review: {str(e)}")
            add_error_comment(
                state, file_path, str(e), "Integration Specialist", "best_practices"
            )

    # Add a summary message
    add_summary_message(
        state,
        "Integration Specialist",
        "integration_specialist",
        "integration and documentation issues",
    )

    return state
