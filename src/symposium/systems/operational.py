# ======= VSM SYSTEM 1: OPERATIONS =======
# Functions that perform specific review tasks
import uuid

from langchain_core.messages import AIMessage
from langchain_ollama import ChatOllama

from symposium.models.base import (
    SeverityLevel,
    CodeLocation,
    ReviewComment,
    CodeReviewState,
)


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
