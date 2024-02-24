import subprocess
from typing import List


# NOTE: This was written by ChatGPT.
def ensure_static_codebase(
    excluded_dirs: List[str] = ["runs", "notebooks"], strict: bool = True
):
    """
    Checks if the current Git repository has a clean working tree and reports
    untracked files, excluding specified directories. In strict mode, raises an
    error if there are modified, staged, or untracked files outside the
    excluded directories.

    Args:
    - excluded_dirs (list of str): Directories to exclude from untracked files
      check, relative to the Git root.
    - strict (bool): If True, raises an error for issues found outside excluded
      directories.

    Returns:
    - None

    Raises:
    - RuntimeError: If strict is True and there are modifications or untracked
      files outside excluded directories.
    """
    try:
        # Find the Git repository root to ensure paths are handled correctly.
        git_root = (
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"], stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
        )

        # Get the status of all files, excluding those in .gitignore
        status_output = (
            subprocess.check_output(
                ["git", "status", "--porcelain=v1"], stderr=subprocess.STDOUT
            )
            .decode()
            .strip()
        )

        # Split the output into lines for individual processing
        status_lines = status_output.split("\n") if status_output else []

        modified_or_staged_files = any(
            line for line in status_lines if not line.startswith("??")
        )
        untracked_files = [line[3:] for line in status_lines if line.startswith("??")]

        # Check untracked files against excluded directories
        untracked_outside_excluded = [
            file
            for file in untracked_files
            if not any(file.startswith(f"{dir}/") for dir in excluded_dirs)
        ]

        if modified_or_staged_files or untracked_outside_excluded:
            messages = []
            if modified_or_staged_files:
                messages.append(
                    "Warning: Your working tree is not clean. There are modified or staged files."
                )
            if untracked_outside_excluded:
                messages.append(
                    "Warning: There are untracked files not being tracked by Git outside the excluded directories:"
                )
                messages.extend([f" - {file}" for file in untracked_outside_excluded])

            full_message = "\n".join(messages)

            if strict:
                raise RuntimeError(full_message)
            else:
                print(full_message)

    except subprocess.CalledProcessError as e:
        error_msg = f"Error checking Git status: {e.output.decode().strip()}"
        if strict:
            raise RuntimeError(error_msg)
        else:
            print(error_msg)
    except Exception as e:
        if strict:
            raise e
        else:
            print(f"Unexpected error: {e}")
