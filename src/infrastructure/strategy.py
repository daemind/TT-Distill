"""
Infrastructure module strategy definitions.

This module defines the abstract interfaces for infrastructure components:
- LLMClient: Communication with LLM servers
- GitManager: Git operations (worktrees, commits, etc.)
- DockerSandbox: Isolated execution in Docker containers

All interfaces follow the SOLID principles and use Python Protocols
with @runtime_checkable for runtime type checking.
"""

from abc import abstractmethod
from collections.abc import AsyncGenerator
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Protocol, runtime_checkable


class LLMProvider(Enum):
    """Supported LLM providers."""

    HUGGING_FACE = "huggingface"
    OPENAI = "openai"
    LOCAL = "local"
    OLLAMA = "ollama"


class LLMTaskType(Enum):
    """Supported LLM task types."""

    CODE_GENERATION = "code_generation"
    CODE_REVIEW = "code_review"
    TEXT_GENERATION = "text_generation"
    SUMMARIZATION = "summarization"
    TRANSLATION = "translation"
    QUESTION_ANSWERING = "question_answering"


@dataclass
class LLMConfig:
    """Configuration for LLM client."""

    provider: LLMProvider
    model_id: str
    api_key: str | None = None
    base_url: str | None = None
    temperature: float = 0.7
    max_tokens: int = 4096
    timeout: int = 120
    extra_headers: dict[str, str] = field(default_factory=dict)


@dataclass
class LLMRequest:
    """Request to LLM server."""

    messages: list[dict[str, str]]
    task_type: LLMTaskType = LLMTaskType.TEXT_GENERATION
    config: LLMConfig | None = None
    stream: bool = False


@dataclass
class LLMResponse:
    """Response from LLM server."""

    content: str
    model_id: str
    tokens_used: int
    finish_reason: str | None = None
    created_at: datetime = field(default_factory=datetime.now)
    metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMClientProtocol(Protocol):
    """Protocol for LLM client communication."""

    @abstractmethod
    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """Send a request to the LLM server."""
        ...

    @abstractmethod
    def send_request_stream(self, request: LLMRequest) -> AsyncGenerator[str, None]:
        """Send a streaming request to the LLM server."""
        ...

    @abstractmethod
    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a model."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the LLM server is healthy."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close the LLM client connection."""
        ...


@dataclass
class GitOperationResult:
    """Result of a Git operation."""

    success: bool
    output: str | None = None
    error_message: str | None = None
    exit_code: int | None = None
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class GitWorktree:
    """Representation of a Git worktree."""

    path: str
    branch: str
    main_repo: str
    created_at: datetime = field(default_factory=datetime.now)
    status: str = "active"


@dataclass
class GitCommit:
    """Representation of a Git commit."""

    commit_hash: str
    message: str
    author: str
    committed_at: datetime
    files_changed: list[str] = field(default_factory=list)
    additions: int = 0
    deletions: int = 0


@runtime_checkable
class GitManagerProtocol(Protocol):
    """Protocol for Git operations."""

    @abstractmethod
    def init_repository(self, path: str) -> GitOperationResult:
        """Initialize a new Git repository."""
        ...

    @abstractmethod
    def clone_repository(
        self, url: str, path: str, branch: str | None = None
    ) -> GitOperationResult:
        """Clone a repository from URL."""
        ...

    @abstractmethod
    def create_worktree(
        self, repo_path: str, worktree_name: str, branch: str | None = None
    ) -> GitWorktree | None:
        """Create a new Git worktree."""
        ...

    @abstractmethod
    def remove_worktree(self, worktree_path: str) -> GitOperationResult:
        """Remove a Git worktree."""
        ...

    @abstractmethod
    def get_worktrees(self, repo_path: str) -> list[GitWorktree]:
        """List all worktrees for a repository."""
        ...

    @abstractmethod
    def add_files(self, repo_path: str, files: list[str]) -> GitOperationResult:
        """Add files to the staging area."""
        ...

    @abstractmethod
    def commit(
        self,
        repo_path: str,
        message: str,
        author_name: str | None = None,
        author_email: str | None = None,
    ) -> GitCommit | None:
        """Create a new commit."""
        ...

    @abstractmethod
    def get_commits(
        self, repo_path: str, limit: int = 10
    ) -> list[GitCommit]:
        """Get recent commits."""
        ...

    @abstractmethod
    def merge_worktree(
        self, worktree_path: str, main_repo: str, branch: str
    ) -> GitOperationResult:
        """Merge worktree changes into main repository."""
        ...

    @abstractmethod
    def close(self) -> None:
        """Close Git manager resources."""
        ...


@dataclass
class DockerContainer:
    """Representation of a Docker container."""

    container_id: str
    image: str
    status: str
    created_at: datetime = field(default_factory=datetime.now)
    ports: dict[str, int] = field(default_factory=dict)
    mounts: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class DockerSandboxConfig:
    """Configuration for Docker sandbox."""

    image: str = "python:3.12-slim"
    working_dir: str = "/workspace"
    timeout: int = 300
    memory_limit: str = "512m"
    cpu_limit: float = 1.0
    network_mode: str = "none"
    volumes: dict[str, str] = field(default_factory=dict)
    environment: dict[str, str] = field(default_factory=dict)


@dataclass
class DockerSandboxResult:
    """Result of sandbox execution."""

    success: bool
    exit_code: int
    stdout: str
    stderr: str
    container_id: str
    execution_time: float
    timestamp: datetime = field(default_factory=datetime.now)


@runtime_checkable
class DockerSandboxProtocol(Protocol):
    """Protocol for Docker sandbox execution."""

    @abstractmethod
    async def create_sandbox(self, config: DockerSandboxConfig) -> DockerContainer:
        """Create a new sandbox container."""
        ...

    @abstractmethod
    async def execute_command(
        self, container_id: str, command: str, timeout: int | None = None  # noqa: ASYNC109
    ) -> DockerSandboxResult:
        """Execute a command in the sandbox."""
        ...

    @abstractmethod
    async def copy_to_sandbox(
        self, container_id: str, source: str, destination: str
    ) -> bool:
        """Copy files to the sandbox."""
        ...

    @abstractmethod
    async def copy_from_sandbox(
        self, container_id: str, source: str, destination: str
    ) -> bool:
        """Copy files from the sandbox."""
        ...

    @abstractmethod
    async def get_logs(self, container_id: str) -> str:
        """Get container logs."""
        ...

    @abstractmethod
    async def stop_sandbox(self, container_id: str) -> bool:
        """Stop a sandbox container."""
        ...

    @abstractmethod
    async def remove_sandbox(self, container_id: str) -> bool:
        """Remove a sandbox container."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if Docker daemon is healthy."""
        ...

    @abstractmethod
    async def close(self) -> None:
        """Close Docker sandbox resources."""
        ...
