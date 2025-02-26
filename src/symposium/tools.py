import subprocess
import tempfile
import os

from langchain.tools import tool


def create_temp_file(content: str, file_extension: str) -> str:
    """Create a temporary file with the given content and return its path."""
    fd, path = tempfile.mkstemp(suffix=file_extension)
    try:
        with os.fdopen(fd, "w") as f:
            f.write(content)
    except:
        os.close(fd)
        raise
    return path


@tool
def run_linter(code: str, language: str) -> str:
    """
    Run a linter on the provided code snippet.

    Args:
        code: The code content to lint
        language: The programming language (python, javascript, etc.)

    Returns:
        Linting results as a string
    """
    if language == "python":
        # Create a temporary Python file
        temp_file = create_temp_file(code, ".py")
        try:
            # Run flake8
            result = subprocess.run(
                ["flake8", "--max-line-length=100", temp_file],
                capture_output=True,
                text=True,
            )

            if result.returncode != 0:
                # Format error messages
                errors = (
                    result.stdout.strip() if result.stdout else result.stderr.strip()
                )
                # Remove temporary file path from error messages
                errors = errors.replace(temp_file, "file.py")
                return f"Linting found issues:\n{errors}"
            else:
                return "No linting issues found."
        except FileNotFoundError:
            return "Error: Linter tool (flake8) not found. Please install with 'pip install flake8'."
        except Exception as e:
            return f"Error running linter: {str(e)}"
        finally:
            # Clean up temporary file
            os.remove(temp_file)

    elif language == "javascript" or language == "typescript":
        # Create a temporary JS file
        temp_file = create_temp_file(code, ".js")
        try:
            # Run ESLint
            result = subprocess.run(
                ["eslint", temp_file], capture_output=True, text=True
            )

            if result.returncode != 0:
                # Format error messages
                errors = (
                    result.stdout.strip() if result.stdout else result.stderr.strip()
                )
                # Remove temporary file path from error messages
                errors = errors.replace(temp_file, "file.js")
                return f"Linting found issues:\n{errors}"
            else:
                return "No linting issues found."
        except FileNotFoundError:
            return "Error: Linter tool (eslint) not found. Please install with 'npm install -g eslint'."
        except Exception as e:
            return f"Error running linter: {str(e)}"
        finally:
            # Clean up temporary file
            os.remove(temp_file)

    else:
        return f"Linting not supported for {language} language."


@tool
def optimize_code(code: str, language: str) -> str:
    """
    Analyze code for optimization opportunities.

    Args:
        code: The code content to optimize
        language: The programming language (python, javascript, etc.)

    Returns:
        Optimization suggestions as a string
    """
    if language == "python":
        # Create a temporary Python file
        temp_file = create_temp_file(code, ".py")
        try:
            # Run pylint for optimization suggestions
            result = subprocess.run(
                ["pylint", "--disable=all", "--enable=R", temp_file],
                capture_output=True,
                text=True,
            )

            # Get the output regardless of return code (pylint returns non-zero even for warnings)
            output = result.stdout.strip() if result.stdout else result.stderr.strip()

            if "rated at" in output and "previous" not in output:
                # No optimization issues found
                return "No optimization opportunities identified."
            else:
                # Format the output to focus on optimization suggestions
                # Remove temporary file path from output
                output = output.replace(temp_file, "file.py")
                return f"Optimization opportunities:\n{output}"
        except FileNotFoundError:
            return "Error: Optimization tool (pylint) not found. Please install with 'pip install pylint'."
        except Exception as e:
            return f"Error analyzing code for optimization: {str(e)}"
        finally:
            # Clean up temporary file
            os.remove(temp_file)

    else:
        return f"Optimization analysis not supported for {language} language."


@tool
def security_scan(code: str, language: str) -> str:
    """
    Scan code for security vulnerabilities.

    Args:
        code: The code content to scan
        language: The programming language (python, javascript, etc.)

    Returns:
        Security scan results as a string
    """
    if language == "python":
        # Create a temporary Python file
        temp_file = create_temp_file(code, ".py")
        try:
            # Run bandit for security scanning
            result = subprocess.run(
                ["bandit", "-r", temp_file], capture_output=True, text=True
            )

            output = result.stdout.strip() if result.stdout else result.stderr.strip()

            if "No issues identified." in output:
                return "No security vulnerabilities found."
            else:
                # Format the output to focus on security issues
                # Remove temporary file path from output
                output = output.replace(temp_file, "file.py")
                return f"Security scan results:\n{output}"
        except FileNotFoundError:
            return "Error: Security scanning tool (bandit) not found. Please install with 'pip install bandit'."
        except Exception as e:
            return f"Error scanning code for security vulnerabilities: {str(e)}"
        finally:
            # Clean up temporary file
            os.remove(temp_file)

    elif language == "javascript" or language == "typescript":
        # For JavaScript, we could integrate with tools like npm audit
        return "Security scanning for JavaScript requires project-level analysis and is not available for code snippets."

    else:
        return f"Security scanning not supported for {language} language."
