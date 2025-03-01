# ======= VSM SYSTEM 1: OPERATIONS =======
# Functions that perform specific review tasks
import uuid
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from pydantic import BaseModel, Field
from typing import List
from langchain_ollama import ChatOllama

from symposium.tools import run_linter
from symposium.models.base import (
    SeverityLevel,
    CodeLocation,
    ReviewComment,
    CodeReviewState,
)

MODEL = "MFDoom/deepseek-r1-tool-calling"


def code_architect_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Code Architect: Focuses on code structure, readability, maintainability, and DRY principles.
    Analyzes code organization, naming conventions, and adherence to software design principles.
    """
    print("CODE ARCHITECT ANALYZING")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Create a wrapper model for a list of ReviewComments
    class ArchitectResult(BaseModel):
        comments: List[ReviewComment] = Field(default_factory=list)

    # Get LLM
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        ArchitectResult, method="json_schema"
    )

    # Process each file
    for file in new_state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a focused prompt for code architecture analysis
        system_prompt = """You are The Code Architect, an expert in software design principles with decades of experience building maintainable systems. Your primary focus is on code structure, readability, and long-term maintainability.

        When reviewing code, carefully analyze:
        1. How well the code is broken down into appropriate-sized, single-responsibility functions and classes
        2. The clarity and descriptiveness of all variable, function, and class names
        3. Whether the code follows DRY principles or contains unnecessary duplication
        4. If the code strikes the right balance between reusability and unnecessary abstraction

        For each issue, create a ReviewComment with the following structure:
        - id: Generate a unique string ID (can be empty, it will be filled in later)
        - severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO
        - location: An object with:
          - file_path: The path to the file being analyzed
          - line_start: The line number where the issue starts
          - line_end: (Optional) The line number where the issue ends
        - message: A clear description of the architecture/design issue
        - suggestion: A specific recommendation to fix the issue, with code example if appropriate
        - source_agent: Use "code_architect"
        - category: The category of the issue (one of "structure", "naming", "duplication", "abstraction", "readability")

        Your feedback style is constructive but direct. You value elegant, readable solutions and aren't afraid to suggest significant refactoring when it would improve the codebase's long-term health.
        """

        # Create the user message with the code
        user_prompt = f"""
        Analyze this {language} code file for architecture and design issues:
        
        File: {file_path}
        
        ```{language}
        {content}
        ```
        
        Return a list of ReviewComment objects for all issues found, focusing on:
        - Function/class size and responsibilities
        - Variable/function naming
        - Code duplication
        - Appropriate abstraction
        - Overall readability and maintainability
        """

        # Set up the conversation with the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Invoke the LLM with structured output
            result = llm_with_structured_output.invoke(messages)

            # Add reviewer comments to the state
            for comment in result.comments:
                # Ensure we have a new UUID
                comment.id = str(uuid.uuid4())

                # Ensure the file path is correctly set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Add to the pull request comments
                new_state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Code Architect review: {str(e)}")
            # Add a fallback comment for tracking the error
            fallback_comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel.LOW,
                location=CodeLocation(file_path=file_path, line_start=1),
                message=f"Error during architecture analysis for {file_path}: {str(e)}",
                suggestion="Please review this file manually for architecture and design issues.",
                source_agent="code_architect_error",
                category="structure",
            )
            new_state.pull_request.comments.append(fallback_comment)

    # Add a summary message
    architect_comments = [
        c for c in new_state.pull_request.comments if c.source_agent == "code_architect"
    ]
    new_state.messages.append(
        AIMessage(
            content=f"The Code Architect completed analysis and identified {len(architect_comments)} architecture and design issues."
        )
    )

    return new_state


def performance_guardian_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Performance Guardian: Focuses on speed, performance, reliability, and scalability.
    Identifies potential bottlenecks, error handling issues, and scalability concerns.
    """
    print("PERFORMANCE GUARDIAN ANALYZING")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Create a wrapper model for a list of ReviewComments
    class PerformanceResult(BaseModel):
        comments: List[ReviewComment] = Field(default_factory=list)

    # Get LLM
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        PerformanceResult, method="json_schema"
    )

    # Process each file
    for file in new_state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a focused prompt for performance analysis
        system_prompt = """You are The Performance Guardian, a systems expert obsessed with efficiency, reliability, and performance at scale. Your role is to ensure code not only works correctly but does so efficiently and reliably under all conditions.

        When analyzing code, focus on:
        1. Potential performance bottlenecks, especially in database queries, API calls, and resource-intensive operations
        2. How the code handles errors and edge cases - is it defensively written to handle failures gracefully?
        3. Whether the code will scale effectively under unexpected load conditions
        4. Appropriate use of caching, lazy loading, and asynchronous processing

        For each issue, create a ReviewComment with the following structure:
        - id: Generate a unique string ID (can be empty, it will be filled in later)
        - severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO
        - location: An object with:
          - file_path: The path to the file being analyzed
          - line_start: The line number where the issue starts
          - line_end: (Optional) The line number where the issue ends
        - message: A clear description of the performance/reliability issue
        - suggestion: A specific recommendation to fix the issue, with code example if appropriate
        - source_agent: Use "performance_guardian"
        - category: The category of the issue (one of "performance", "error_handling", "scalability", "resource_usage", "reliability")

        Your feedback style is data-driven and pragmatic. Prioritize your recommendations based on expected user impact.
        """

        # Create the user message with the code
        user_prompt = f"""
        Analyze this {language} code file for performance and reliability issues:
        
        File: {file_path}
        
        ```{language}
        {content}
        ```
        
        Return a list of ReviewComment objects for all issues found, focusing on:
        - Performance bottlenecks
        - Error handling and edge cases
        - Scalability concerns
        - Resource usage efficiency
        - Overall reliability under stress
        """

        # Set up the conversation with the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Invoke the LLM with structured output
            result = llm_with_structured_output.invoke(messages)

            # Add reviewer comments to the state
            for comment in result.comments:
                # Ensure we have a new UUID
                comment.id = str(uuid.uuid4())

                # Ensure the file path is correctly set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Add to the pull request comments
                new_state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Performance Guardian review: {str(e)}")
            # Add a fallback comment for tracking the error
            fallback_comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel.LOW,
                location=CodeLocation(file_path=file_path, line_start=1),
                message=f"Error during performance analysis for {file_path}: {str(e)}",
                suggestion="Please review this file manually for performance and reliability issues.",
                source_agent="performance_guardian_error",
                category="performance",
            )
            new_state.pull_request.comments.append(fallback_comment)

    # Add a summary message
    performance_comments = [
        c
        for c in new_state.pull_request.comments
        if c.source_agent == "performance_guardian"
    ]
    new_state.messages.append(
        AIMessage(
            content=f"The Performance Guardian completed analysis and identified {len(performance_comments)} performance and reliability issues."
        )
    )

    return new_state


def security_sentinel_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Security Sentinel: Focuses on security vulnerabilities and test coverage.
    Identifies potential attack vectors and ensures adequate test coverage for critical functionality.
    """
    print("SECURITY SENTINEL ANALYZING")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Create a wrapper model for a list of ReviewComments
    class SecurityResult(BaseModel):
        comments: List[ReviewComment] = Field(default_factory=list)

    # Get LLM
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        SecurityResult, method="json_schema"
    )

    # Process each file
    for file in new_state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a focused prompt for security analysis
        system_prompt = """You are The Security Sentinel, a cybersecurity expert and testing advocate with a background in penetration testing and defensive programming. Your mission is to identify security vulnerabilities and ensure code is properly tested against potential attack vectors.

        When reviewing code, thoroughly investigate:
        1. Potential security vulnerabilities, including injection attacks, authentication issues, and data exposure
        2. The quality and coverage of tests, especially for security-critical functionality
        3. Edge cases that might not be covered by existing tests
        4. Potential misuse scenarios - put yourself in the mindset of someone trying to exploit the system

        For each issue, create a ReviewComment with the following structure:
        - id: Generate a unique string ID (can be empty, it will be filled in later)
        - severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO
        - location: An object with:
          - file_path: The path to the file being analyzed
          - line_start: The line number where the issue starts
          - line_end: (Optional) The line number where the issue ends
        - message: A clear description of the security vulnerability or testing gap
        - suggestion: A specific recommendation to fix the issue, with code example if appropriate
        - source_agent: Use "security_sentinel"
        - category: The category of the issue (one of "security", "authentication", "data_exposure", "test_coverage", "input_validation")

        Your feedback style is thorough and methodical. Examine code from an adversarial perspective, asking "How could this be exploited?" at every step.
        """

        # Create the user message with the code
        user_prompt = f"""
        Analyze this {language} code file for security vulnerabilities and testing gaps:
        
        File: {file_path}
        
        ```{language}
        {content}
        ```
        
        Return a list of ReviewComment objects for all issues found, focusing on:
        - Security vulnerabilities
        - Authentication and authorization issues
        - Data exposure risks
        - Testing gaps and edge cases
        - Input validation concerns
        """

        # Set up the conversation with the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Invoke the LLM with structured output
            result = llm_with_structured_output.invoke(messages)

            # Add reviewer comments to the state
            for comment in result.comments:
                # Ensure we have a new UUID
                comment.id = str(uuid.uuid4())

                # Ensure the file path is correctly set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Add to the pull request comments
                new_state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Security Sentinel review: {str(e)}")
            # Add a fallback comment for tracking the error
            fallback_comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel.LOW,
                location=CodeLocation(file_path=file_path, line_start=1),
                message=f"Error during security analysis for {file_path}: {str(e)}",
                suggestion="Please review this file manually for security vulnerabilities and testing gaps.",
                source_agent="security_sentinel_error",
                category="security",
            )
            new_state.pull_request.comments.append(fallback_comment)

    # Add a summary message
    security_comments = [
        c
        for c in new_state.pull_request.comments
        if c.source_agent == "security_sentinel"
    ]
    new_state.messages.append(
        AIMessage(
            content=f"The Security Sentinel completed analysis and identified {len(security_comments)} security vulnerabilities and testing gaps."
        )
    )

    return new_state


def integration_specialist_review(state: CodeReviewState) -> CodeReviewState:
    """
    The Integration Specialist: Focuses on documentation, appropriate use of language features,
    avoiding reinventing the wheel, and architectural coherence.
    """
    print("INTEGRATION SPECIALIST ANALYZING")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Create a wrapper model for a list of ReviewComments
    class IntegrationResult(BaseModel):
        comments: List[ReviewComment] = Field(default_factory=list)

    # Get LLM
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        IntegrationResult, method="json_schema"
    )

    # Process each file
    for file in new_state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Create a focused prompt for integration analysis
        system_prompt = """You are The Integration Specialist, a polyglot developer with extensive knowledge across multiple programming languages, frameworks, and architectural patterns. Your expertise lies in ensuring code leverages appropriate language features, follows best practices, and integrates seamlessly with the broader system.

        When reviewing code, focus on:
        1. Documentation completeness and accuracy, including READMEs, inline comments, and API documentation
        2. Whether the code unnecessarily reinvents functionality available in standard libraries or frameworks
        3. Appropriate use of language-specific idioms and features
        4. How well the code aligns with the project's overall architecture and technology choices

        For each issue, create a ReviewComment with the following structure:
        - id: Generate a unique string ID (can be empty, it will be filled in later)
        - severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO
        - location: An object with:
          - file_path: The path to the file being analyzed
          - line_start: The line number where the issue starts
          - line_end: (Optional) The line number where the issue ends
        - message: A clear description of the integration/best practice issue
        - suggestion: A specific recommendation to fix the issue, with code example if appropriate
        - source_agent: Use "integration_specialist"
        - category: The category of the issue (one of "documentation", "reinventing_wheel", "language_usage", "architectural_alignment", "best_practices")

        Your feedback style is pragmatic and knowledge-sharing. You understand that developers may not be aware of all available tools and patterns, so you emphasize education alongside critique.
        """

        # Create the user message with the code
        user_prompt = f"""
        Analyze this {language} code file for integration and best practice issues:
        
        File: {file_path}
        
        ```{language}
        {content}
        ```
        
        Return a list of ReviewComment objects for all issues found, focusing on:
        - Documentation completeness and quality
        - Code that reinvents existing functionality
        - Appropriate use of language features
        - Alignment with architectural best practices
        - Overall adherence to language-specific conventions
        """

        # Set up the conversation with the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Invoke the LLM with structured output
            result = llm_with_structured_output.invoke(messages)

            # Add reviewer comments to the state
            for comment in result.comments:
                # Ensure we have a new UUID
                comment.id = str(uuid.uuid4())

                # Ensure the file path is correctly set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Add to the pull request comments
                new_state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in Integration Specialist review: {str(e)}")
            # Add a fallback comment for tracking the error
            fallback_comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel.LOW,
                location=CodeLocation(file_path=file_path, line_start=1),
                message=f"Error during integration analysis for {file_path}: {str(e)}",
                suggestion="Please review this file manually for integration and best practice issues.",
                source_agent="integration_specialist_error",
                category="best_practices",
            )
            new_state.pull_request.comments.append(fallback_comment)

    # Add a summary message
    integration_comments = [
        c
        for c in new_state.pull_request.comments
        if c.source_agent == "integration_specialist"
    ]
    new_state.messages.append(
        AIMessage(
            content=f"The Integration Specialist completed analysis and identified {len(integration_comments)} integration and best practice issues."
        )
    )

    return new_state


def system1_lint_code(state: CodeReviewState) -> CodeReviewState:
    """
    VSM System 1: Operation
    Lints the code files using an external linting tool and passes the results to an LLM.
    Runs the linter directly and incorporates the output into the prompt for the LLM.
    Uses structured output to generate ReviewComment objects directly.
    """
    print("LINT")
    new_state = state.model_copy(deep=True)
    if not new_state.pull_request:
        return new_state

    # Create a wrapper model for a list of ReviewComments
    class LintingResult(BaseModel):
        comments: List[ReviewComment] = Field(default_factory=list)

    # Get LLM
    llm = ChatOllama(model=MODEL)
    llm_with_structured_output = llm.with_structured_output(
        LintingResult, method="json_schema"
    )

    # Process each file
    for file in new_state.pull_request.files:
        file_path = file.metadata.path
        content = file.content
        language = file.metadata.language or ""

        # Run the linter directly to get results
        try:
            linter_results = run_linter.func(code=content, language=language)
        except Exception as e:
            print(f"Error running linter: {str(e)}")
            linter_results = f"Error running linter: {str(e)}"

        # Create a focused prompt for code style analysis that includes linter output
        system_prompt = """You are an expert code reviewer, you specialize in linting-focused comments, focusing on code style and best practices.
        
        You will be provided with the output of an automated linting tool along with the code to analyze.
        
        Your process:
        1. Review the linting tool output provided
        2. Analyze the code yourself to find additional style issues
        3. Focus on naming conventions, indentation, code organization, comments, and language-specific best practices
        
        For each issue, create a ReviewComment with the following structure:
        - id: Generate a unique string ID (can be empty, it will be filled in later)
        - severity: One of CRITICAL, HIGH, MEDIUM, LOW, or INFO
        - location: An object with:
          - file_path: The path to the file being analyzed
          - line_start: The line number where the issue starts
          - line_end: (Optional) The line number where the issue ends
        - message: A clear description of the style issue
        - suggestion: A specific recommendation to fix the issue, even if it is obvious.
        - source_agent: Use "linter"
        - category: The category of the issue (one of "style", "naming", "complexity", "documentation" or "best practices")

        Create as many ReviewComments as needed. 
        Review style guide:
         - You may use "~" as a prefix for issues that are minor and could be ignored ("~lines longer than 80 characters.")
         - You may use "~~" as a prefix for issues that are very minor likely will be ignored ("~~ no newline at end of file.")
        """

        # Create the user message with the code and linter output
        user_prompt = f"""
        Analyze this {language} code file for style issues:
        
        File: {file_path}
        
        ```{language}
        {content}
        ```
        
        The automated linter tool produced the following output:
        ```
        {linter_results}
        ```
        
        Review the linter output and enhance with your own style analysis.
        Return a list of ReviewComment objects for all issues found, one ReviewComment per issue.
        """

        # Set up the conversation with the LLM
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ]

        try:
            # Invoke the LLM with structured output
            result = llm_with_structured_output.invoke(messages)

            __import__("ipdb").set_trace()
            # Add reviewer comments to the state
            for comment in result.comments:
                # Ensure we have a new UUID
                comment.id = str(uuid.uuid4())

                # Ensure the file path is correctly set
                if hasattr(comment.location, "file_path"):
                    comment.location.file_path = file_path

                # Add to the pull request comments
                new_state.pull_request.comments.append(comment)

        except Exception as e:
            print(f"Error in LLM-based linting: {str(e)}")
            # Add a fallback comment for tracking the error
            fallback_comment = ReviewComment(
                id=str(uuid.uuid4()),
                severity=SeverityLevel.LOW,
                location=CodeLocation(file_path=file_path, line_start=1),
                message=f"Error during style analysis for {file_path}: {str(e)}",
                suggestion="Please review this file manually for style issues.",
                source_agent="linter_error",
                category="style",
            )
            new_state.pull_request.comments.append(fallback_comment)

    # Add a summary message
    new_state.messages.append(
        AIMessage(
            content=f"Performed code style analysis and found {len(new_state.pull_request.comments)} style issues."
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
