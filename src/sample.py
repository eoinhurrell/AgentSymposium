from symposium.models.base import (
    CodeReviewState,
    PullRequest,
    PullRequestMetadata,
    CodeFile,
    FileMetadata,
)
from symposium.vsm.operational import system1_lint_code

from langchain_core.messages import HumanMessage, SystemMessage


def main():
    pull_request = PullRequest(
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
    code_state = CodeReviewState(
        pull_request=pull_request,
        messages=[
            SystemMessage(content="Code review agent analyzing pull request."),
            HumanMessage(
                content=f"Please review PR #{pull_request.metadata.id}: {pull_request.metadata.title}"
            ),
        ],
        # context={"llm": llm},
    )
    print("Created state, running")
    new_state = system1_lint_code(code_state)
    __import__("ipdb").set_trace()
    pass


if __name__ == "__main__":
    main()
