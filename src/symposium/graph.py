from langgraph.graph import END, StateGraph

from symposium.models.base import (
    CodeReviewState,
)
from symposium.vsm.control import system3_control_review_process
from symposium.vsm.coordination import system2_coordinate_reviews
from symposium.vsm.intelligence import system4_analyze_context
from symposium.vsm.operational import (
    code_architect_review,
    performance_guardian_review,
    security_sentinel_review,
    integration_specialist_review,
)
from symposium.vsm.policy import system5_make_review_decision


def create_code_review_agent():
    """
    Creates a LangGraph agent for code review based on the Viable Systems Model.

    The agent follows the VSM structure with specialized personas for System 1:
    - System 1 (Operations): Code Architect, Performance Guardian, Security Sentinel, Integration Specialist
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
    # System 1: Operations with specialized personas
    workflow.add_node("code_architect", code_architect_review)
    workflow.add_node("performance_guardian", performance_guardian_review)
    workflow.add_node("security_sentinel", security_sentinel_review)
    workflow.add_node("integration_specialist", integration_specialist_review)

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
            "code_architect": "code_architect",
            "performance_guardian": "performance_guardian",
            "security_sentinel": "security_sentinel",
            "integration_specialist": "integration_specialist",
            "coordinator": "coordinate_reviews",
            "intelligence": "analyze_context",
            "policy": "make_review_decision",
            END: END,  # Handle END case explicitly
        },
    )

    # All operations (System 1) go back to control
    workflow.add_edge("code_architect", "control_review_process")
    workflow.add_edge("performance_guardian", "control_review_process")
    workflow.add_edge("security_sentinel", "control_review_process")
    workflow.add_edge("integration_specialist", "control_review_process")

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
