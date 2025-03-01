from typing import Optional

# LangChain imports
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_ollama import ChatOllama

# LangGraph imports
from langgraph.graph import StateGraph, END

from symposium.models.base import (
    CodeFile,
    FileMetadata,
    PullRequest,
    PullRequestMetadata,
    CodeReviewState,
)
from symposium.vsm.operational import (
    system1_complexity_analysis,
    system1_lint_code,
    system1_security_check,
)
from symposium.vsm.coordination import system2_coordinate_reviews
from symposium.vsm.control import system3_control_review_process
from symposium.vsm.intelligence import system4_analyze_context
from symposium.vsm.policy import system5_make_review_decision


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
