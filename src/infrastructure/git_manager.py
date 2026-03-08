# ruff: noqa: S603
"""
Git Manager implementation for the Cybernetic Production Studio.

This module provides Git operations for the Cybernetic Production Studio:
- Repository initialization and cloning
- Git worktree management for isolated agent execution
- Commit operations with metadata tracking
- Branch management

The implementation follows the GitManagerProtocol and ensures:
- Physical isolation of agent code via worktrees
- Atomic commits with proper metadata
- Safe merge operations
"""

import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.infrastructure.strategy import (
    GitCommit,
    GitManagerProtocol,
    GitOperationResult,
    GitWorktree,
)

logger = logging.getLogger(__name__)

# Constants for subprocess security
_GIT_CMD = "git"


class GitManager(GitManagerProtocol):
    """Implementation of Git manager for repository operations."""

    def __init__(self, base_path: str | Path | None = None) -> None:
        """Initialize the Git manager.

        Args:
            base_path: Base path for Git operations. Defaults to current directory.
        """
        self._base_path = Path(base_path) if base_path else Path.cwd()
        self._worktrees: dict[str, GitWorktree] = {}

    def init_repository(self, path: str) -> GitOperationResult:
        """Initialize a new Git repository.

        Args:
            path: Path where to initialize the repository.

        Returns:
            GitOperationResult with success status and metadata.
        """
        repo_path = Path(path)
        repo_path.mkdir(parents=True, exist_ok=True)

        try:
            subprocess.run(
                [_GIT_CMD, "init"],

                cwd=str(repo_path),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            logger.info(f"Initialized Git repository at {repo_path}")

            return GitOperationResult(
                success=True,
                output="",
                error_message=None,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to initialize repository: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error initializing repository: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )

    def clone_repository(
        self, url: str, path: str, branch: str | None = None
    ) -> GitOperationResult:
        """Clone a repository from URL.

        Args:
            url: Repository URL to clone.
            path: Local path for the cloned repository.
            branch: Optional branch to clone.

        Returns:
            GitOperationResult with success status and metadata.
        """
        clone_args = [_GIT_CMD, "clone", url, str(path)]

        if branch:
            clone_args.extend(["--branch", branch, "--single-branch"])

        try:
            subprocess.run(
                clone_args,
                capture_output=True,
                text=True,
                check=True,
                shell=False,

            )

            logger.info(f"Cloned repository {url} to {path}")

            return GitOperationResult(
                success=True,
                output="",
                error_message=None,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone repository: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error cloning repository: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )

    def create_worktree(
        self, repo_path: str, worktree_name: str, branch: str | None = None
    ) -> GitWorktree | None:
        """Create a new Git worktree.

        Args:
            repo_path: Path to the main repository.
            worktree_name: Name for the worktree (also used as branch name).
            branch: Optional base branch to create worktree from.

        Returns:
            GitWorktree if successful, None otherwise.
        """
        repo = Path(repo_path)
        worktree_dir = repo / f"worktree-{worktree_name}"

        try:
            # Create worktree
            args = [_GIT_CMD, "worktree", "add", str(worktree_dir)]

            if branch:
                args.extend(["-b", worktree_name, branch])
            else:
                args.extend(["-b", worktree_name])

            subprocess.run(
                args,

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            logger.info(f"Created worktree {worktree_name} at {worktree_dir}")

            # Get worktree info
            branch_result = subprocess.run(
                [_GIT_CMD, "branch", "--show-current"],

                cwd=str(worktree_dir),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            current_branch = branch_result.stdout.strip()

            worktree = GitWorktree(
                path=str(worktree_dir),
                branch=current_branch,
                main_repo=str(repo),
                status="active",
            )

            self._worktrees[worktree_name] = worktree

            return worktree
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create worktree: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error creating worktree: {e}")
            return None

    def merge_worktree(
        self, worktree_path: str, main_repo: str, branch: str
    ) -> GitOperationResult:
        """Merge worktree changes into main repository.

        Args:
            worktree_path: Path to the worktree.
            main_repo: Path to the main repository.
            branch: Branch to merge into.

        Returns:
            GitOperationResult with success status and metadata.
        """
        worktree = Path(worktree_path)
        repo = Path(main_repo)

        try:
            # Fetch changes from worktree
            subprocess.run(
                [_GIT_CMD, "fetch", str(worktree)],
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )

            # Merge changes
            merge_args = [_GIT_CMD, "merge", f"refs/remotes/worktree/{worktree.name}"]

            result = subprocess.run(
                merge_args,

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=False,
                shell=False,
            )

            if result.returncode == 0:
                logger.info(f"Merged worktree changes into {branch}")
                return GitOperationResult(
                    success=True,
                    output=result.stdout,
                    error_message=None,
                    exit_code=None,
                )
            logger.warning(f"Merge completed with warnings: {result.stderr}")
            return GitOperationResult(
                success=True,
                output=result.stdout,
                error_message=result.stderr,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to merge worktree: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error merging worktree: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )

    def commit_changes(
        self,
        repo_path: str,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> GitCommit | None:
        """Commit changes in a repository.

        Args:
            repo_path: Path to the repository.
            message: Commit message.
            author_name: Optional author name.
            author_email: Optional author email.

        Returns:
            GitCommit if successful, None otherwise.
        """
        repo = Path(repo_path)

        try:
            # Stage all changes
            subprocess.run(
                [_GIT_CMD, "add", "."],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            # Commit
            commit_args = [_GIT_CMD, "commit", "-m", message]

            if author_name and author_email:
                commit_args.extend(["--author", f"{author_name} <{author_email}>"])

            subprocess.run(
                commit_args,

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            # Get commit hash
            hash_result = subprocess.run(
                [_GIT_CMD, "rev-parse", "HEAD"],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            commit_hash = hash_result.stdout.strip()

            commit = GitCommit(
                commit_hash=commit_hash,
                message=message,
                author=author_name or "",
                committed_at=datetime.now(UTC),
                files_changed=[],
                additions=0,
                deletions=0,
            )

            logger.info(f"Committed changes: {commit_hash[:8]} - {message}")

            return commit
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to commit changes: {e.stderr}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error committing changes: {e}")
            return None

    def get_status(self, repo_path: str) -> dict[str, Any]:
        """Get Git status for a repository.

        Args:
            repo_path: Path to the repository.

        Returns:
            Dictionary with status information.
        """
        repo = Path(repo_path)

        try:
            # Get short status
            status_result = subprocess.run(
                [_GIT_CMD, "status", "--short"],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            # Get current branch
            branch_result = subprocess.run(
                [_GIT_CMD, "branch", "--show-current"],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            current_branch = branch_result.stdout.strip()

            # Parse status lines
            files = []
            for line in status_result.stdout.strip().split("\n"):
                if line:
                    parts = line.split(" ", 1)
                    if len(parts) == 2:
                        status, filename = parts
                        files.append({"status": status, "filename": filename})

            return {
                "current_branch": current_branch,
                "is_clean": len(files) == 0,
                "files": files,
            }
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get status: {e.stderr}")
            return {"error": e.stderr}
        except Exception as e:
            logger.error(f"Unexpected error getting status: {e}")
            return {"error": str(e)}

    def list_worktrees(self, repo_path: str) -> list[GitWorktree]:
        """List all worktrees for a repository.

        Args:
            repo_path: Path to the repository.

        Returns:
            List of GitWorktree objects.
        """
        repo = Path(repo_path)

        try:
            result = subprocess.run(
                [_GIT_CMD, "worktree", "list", "--porcelain"],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            worktrees = []
            current_worktree: dict[str, str] = {}

            for line in result.stdout.strip().split("\n"):
                if line.startswith("worktree"):
                    if current_worktree:
                        worktrees.append(self._parse_worktree_info(current_worktree))
                    current_worktree = {"worktree": line.split(" ", 1)[1]}
                elif line.startswith("branch"):
                    current_worktree["branch"] = line.split(" ", 1)[1]
                elif line.startswith("HEAD"):
                    current_worktree["head"] = line.split(" ", 1)[1]

            if current_worktree:
                worktrees.append(self._parse_worktree_info(current_worktree))

            return worktrees
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to list worktrees: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error listing worktrees: {e}")
            return []

    def delete_worktree(self, worktree_path: str, force: bool = False) -> bool:
        """Delete a worktree.

        Args:
            worktree_path: Path to the worktree.
            force: Force deletion even if changes exist.

        Returns:
            True if successful, False otherwise.
        """
        worktree = Path(worktree_path)

        try:
            args = [_GIT_CMD, "worktree", "remove", str(worktree)]

            if force:
                args.append("--force")

            subprocess.run(
                args,

                cwd=str(worktree.parent),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            worktree_name = worktree.name
            if worktree_name in self._worktrees:
                del self._worktrees[worktree_name]

            logger.info(f"Deleted worktree: {worktree_path}")

            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to delete worktree: {e.stderr}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error deleting worktree: {e}")
            return False

    def add_files(self, repo_path: str, files: list[str]) -> GitOperationResult:
        """Add files to the staging area.

        Args:
            repo_path: Path to the repository.
            files: List of files to add.

        Returns:
            GitOperationResult with success status and metadata.
        """
        repo = Path(repo_path)

        try:
            subprocess.run(
                [_GIT_CMD, "add", *files],
                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            logger.info(f"Added files: {files}")

            return GitOperationResult(
                success=True,
                output="",
                error_message=None,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to add files: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error adding files: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )

    def commit(
        self,
        repo_path: str,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> GitCommit | None:
        """Create a new commit.

        Args:
            repo_path: Path to the repository.
            message: Commit message.
            author_name: Optional author name.
            author_email: Optional author email.

        Returns:
            GitCommit if successful, None otherwise.
        """
        return self.commit_changes(repo_path, message, author_name, author_email)

    def get_commits(self, repo_path: str, limit: int = 10) -> list[GitCommit]:
        """Get recent commits.

        Args:
            repo_path: Path to the repository.
            limit: Maximum number of commits to return.

        Returns:
            List of GitCommit objects.
        """
        return self.get_commit_history(repo_path, limit)

    def get_worktrees(self, repo_path: str) -> list[GitWorktree]:
        """List all worktrees for a repository.

        Args:
            repo_path: Path to the repository.

        Returns:
            List of GitWorktree objects.
        """
        return self.list_worktrees(repo_path)

    def remove_worktree(self, worktree_path: str) -> GitOperationResult:
        """Remove a Git worktree.

        Args:
            worktree_path: Path to the worktree.

        Returns:
            GitOperationResult with success status and metadata.
        """
        success = self.delete_worktree(worktree_path, force=False)

        if not success:
            return GitOperationResult(
                success=False,
                output="",
                error_message="Failed to delete worktree",
                exit_code=None,
            )

        return GitOperationResult(
            success=True,
            output="",
            error_message=None,
            exit_code=None,
        )

    def close(self) -> None:
        """Close Git manager resources."""
        # Clean up any remaining worktrees
        for worktree_name, worktree in list(self._worktrees.items()):
            try:
                self.delete_worktree(worktree.path, force=True)
            except Exception as e:
                logger.warning(f"Failed to clean up worktree {worktree_name}: {e}")

        self._worktrees.clear()
        logger.info("Git manager closed")

    def _parse_worktree_info(self, info: dict[str, str]) -> GitWorktree:
        """Parse worktree information from subprocess output."""
        worktree_path = info.get("worktree", "")
        branch = info.get("branch", "").replace("refs/heads/", "")

        return GitWorktree(
            path=worktree_path,
            branch=branch,
            main_repo=str(Path(worktree_path).parent),
            status="active",
        )

    def get_commit_history(
        self, repo_path: str, limit: int = 10
    ) -> list[GitCommit]:
        """Get commit history for a repository.

        Args:
            repo_path: Path to the repository.
            limit: Maximum number of commits to return.

        Returns:
            List of GitCommit objects.
        """
        repo = Path(repo_path)

        try:
            log_format = (
                "%H|%s|%an|%ae|%aI"
            )  # hash|subject|author_name|author_email|date
            result = subprocess.run(
                [_GIT_CMD, "log", f"-{limit}", "--pretty=format:" + log_format],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            commits = []
            for line in result.stdout.strip().split("\n"):
                if line:
                    parts = line.split("|")
                    if len(parts) == 5:
                        commits.append(
                            GitCommit(
                                commit_hash=parts[0],
                                message=parts[1],
                                author=parts[2],
                                committed_at=datetime.fromisoformat(
                                    parts[4].replace("Z", "+00:00")
                                ),
                                files_changed=[],
                                additions=0,
                                deletions=0,
                            )
                        )

            return commits
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to get commit history: {e.stderr}")
            return []
        except Exception as e:
            logger.error(f"Unexpected error getting commit history: {e}")
            return []

    def checkout_branch(self, repo_path: str, branch: str) -> GitOperationResult:
        """Checkout a branch in a repository.

        Args:
            repo_path: Path to the repository.
            branch: Branch to checkout.

        Returns:
            GitOperationResult with success status and metadata.
        """
        repo = Path(repo_path)

        try:
            subprocess.run(
                [_GIT_CMD, "checkout", branch],

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            logger.info(f"Checked out branch: {branch}")

            return GitOperationResult(
                success=True,
                output="",
                error_message=None,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to checkout branch: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error checking out branch: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )

    def create_branch(
        self, repo_path: str, branch_name: str, base_branch: str | None = None
    ) -> GitOperationResult:
        """Create a new branch in a repository.

        Args:
            repo_path: Path to the repository.
            branch_name: Name for the new branch.
            base_branch: Optional base branch to create from.

        Returns:
            GitOperationResult with success status and metadata.
        """
        repo = Path(repo_path)

        try:
            args = [_GIT_CMD, "checkout", "-b", branch_name]

            if base_branch:
                args.insert(2, base_branch)

            subprocess.run(
                args,

                cwd=str(repo),
                capture_output=True,
                text=True,
                check=True,
                shell=False,
            )

            logger.info(f"Created branch: {branch_name}")

            return GitOperationResult(
                success=True,
                output="",
                error_message=None,
                exit_code=None,
            )
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to create branch: {e.stderr}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=e.stderr,
                exit_code=None,
            )
        except Exception as e:
            logger.error(f"Unexpected error creating branch: {e}")
            return GitOperationResult(
                success=False,
                output="",
                error_message=str(e),
                exit_code=None,
            )
