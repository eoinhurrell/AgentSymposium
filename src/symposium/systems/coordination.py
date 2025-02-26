# ======= VSM SYSTEM 2: COORDINATION =======
# Coordinates the operational units and resolves conflicts

from langchain_core.messages import AIMessage

from symposium.models.base import (
    SeverityLevel,
    CodeReviewState,
)


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
