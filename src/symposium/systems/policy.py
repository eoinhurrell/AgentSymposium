# ======= VSM SYSTEM 5: POLICY =======
# Makes high-level decisions and sets priorities

# LangChain imports
from langchain_core.messages import AIMessage

# LangGraph imports

from symposium.models.base import (
    SeverityLevel,
    CodeReviewState,
)


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
