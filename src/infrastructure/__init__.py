"""
Infrastructure module for the Cybernetic Production Studio.

This module provides low-level infrastructure components:
- LLMClient: Communication with LLM servers
- GitManager: Git operations (worktrees, commits, etc.)
- DockerSandbox: Isolated execution in Docker containers

All components follow the SOLID principles and use Python Protocols
with @runtime_checkable for runtime type checking.
"""

from src.infrastructure.docker_sandbox import DockerSandbox
from src.infrastructure.git_manager import GitManager
from src.infrastructure.llm_client import LLMClient
from src.infrastructure.strategy import (
    DockerContainer,
    DockerSandboxConfig,
    DockerSandboxProtocol,
    DockerSandboxResult,
    GitCommit,
    GitManagerProtocol,
    GitOperationResult,
    GitWorktree,
    LLMClientProtocol,
    LLMConfig,
    LLMProvider,
    LLMRequest,
    LLMResponse,
    LLMTaskType,
)

__all__ = [
    "DockerContainer",
    "DockerSandbox",
    "DockerSandboxConfig",
    "DockerSandboxProtocol",
    "DockerSandboxResult",
    "GitCommit",
    "GitManager",
    "GitManagerProtocol",
    "GitOperationResult",
    "GitWorktree",
    "LLMClient",
    "LLMClientProtocol",
    "LLMConfig",
    "LLMProvider",
    "LLMRequest",
    "LLMResponse",
    "LLMTaskType",
]
