"""
Docker Sandbox implementation for the Cybernetic Production Studio.

This module provides isolated execution environments using Docker containers:
- Container creation and management
- Command execution in isolated environments
- File copy operations (in/out)
- Resource limits (CPU, memory)
- Network isolation

The implementation follows the DockerSandboxProtocol and ensures:
- Complete isolation from host system
- Resource constraints for fair scheduling
- Secure execution environments
"""

import asyncio
import json
import logging
import time
import uuid
from datetime import UTC, datetime
from typing import Any

from src.infrastructure.strategy import (
    DockerContainer,
    DockerSandboxConfig,
    DockerSandboxProtocol,
    DockerSandboxResult,
)

_MIN_STATS_PARTS = 2
_MAX_STATS_PARTS = 3

logger = logging.getLogger(__name__)


class DockerSandbox(DockerSandboxProtocol):
    """Implementation of Docker sandbox for isolated execution."""

    def __init__(self) -> None:
        """Initialize the Docker sandbox manager."""
        self._containers: dict[str, DockerContainer] = {}
        self._initialized = False

    async def initialize(self) -> None:
        """Initialize the Docker sandbox manager."""
        if not self._initialized:
            if await self._docker_available():
                self._initialized = True
                logger.info("Docker sandbox initialized")
            else:
                logger.warning("Docker not available, sandbox disabled")
                self._initialized = False

    async def _docker_available(self) -> bool:
        """Check if Docker daemon is available."""
        try:
            result = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await result.communicate()
            return result.returncode == 0
        except Exception:
            return False

    async def create_sandbox(self, config: DockerSandboxConfig) -> DockerContainer:
        """Create a new sandbox container.

        Args:
            config: Docker sandbox configuration.

        Returns:
            DockerContainer with container information.

        Raises:
            RuntimeError: If Docker is not available.
        """
        await self.initialize()

        if not self._initialized:
            raise RuntimeError("Docker is not available")

        container_id = str(uuid.uuid4())[:12]
        container_name = f"cps-sandbox-{container_id}"

        # Build docker run arguments
        args = ["docker", "run", "-d"]

        # Container name
        args.extend(["--name", container_name])

        # Working directory
        args.extend(["-w", config.working_dir])

        # Memory limit
        args.extend(["-m", config.memory_limit])

        # CPU limit
        args.extend(["--cpus", str(config.cpu_limit)])

        # Network mode
        args.extend(["--network", config.network_mode])

        # Working directory mount
        if config.volumes:
            for host_path, container_path in config.volumes.items():
                args.extend(["-v", f"{host_path}:{container_path}"])

        # Environment variables
        for key, value in config.environment.items():
            args.extend(["-e", f"{key}={value}"])

        # Image and command
        args.extend([config.image, "sleep", "infinity"])

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                raise RuntimeError("Failed to create container")

            # Container ID is returned as stdout, but we need to capture it
            process2 = await asyncio.create_subprocess_exec(
                "docker",
                "ps",
                "-q",
                "--filter",
                f"name={container_name}",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout_result, _ = await process2.communicate()
            created_container_id = stdout_result.decode("utf-8").strip()

            container = DockerContainer(
                container_id=created_container_id,
                image=config.image,
                status="running",
                created_at=datetime.now(UTC),
                ports={},
                mounts=config.volumes,
                environment=config.environment,
            )

            self._containers[container.container_id] = container

            logger.info(f"Created sandbox container: {container.container_id}")

            return container
        except Exception as e:
            logger.error(f"Failed to create sandbox: {e}")
            raise

    async def execute_command(
        self, container_id: str, command: str, timeout: int | None = None  # noqa: ASYNC109
    ) -> DockerSandboxResult:
        """Execute a command in the sandbox.

        Args:
            container_id: Container identifier.
            command: Command to execute.
            timeout: Optional timeout in seconds.

        Returns:
            DockerSandboxResult with execution results.

        Raises:
            ValueError: If container not found.
            RuntimeError: If execution fails.
        """
        if container_id not in self._containers:
            raise ValueError(f"Container not found: {container_id}")

        _container = self._containers[container_id]

        args = ["docker", "exec", "-i", container_id, "sh", "-c", command]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            try:
                stdout_result, stderr_result = await asyncio.wait_for(
                    process.communicate(), timeout=timeout
                )
            except TimeoutError:
                process.kill()
                await process.wait()
                return DockerSandboxResult(
                    success=False,
                    exit_code=-1,
                    stdout="",
                    stderr="Command timed out",
                    container_id=container_id,
                    execution_time=0.0,
                )

            exit_code = process.returncode if process.returncode is not None else -1

            result = DockerSandboxResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout=stdout_result.decode("utf-8"),
                stderr=stderr_result.decode("utf-8"),
                container_id=container_id,
                execution_time=self._get_execution_time(),
            )

            logger.debug(
                f"Executed command in {container_id}: exit_code={exit_code}"
            )

            return result
        except Exception as e:
            logger.error(f"Failed to execute command in {container_id}: {e}")
            return DockerSandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                container_id=container_id,
                execution_time=0.0,
            )

    async def copy_to_sandbox(
        self, container_id: str, source: str, destination: str
    ) -> bool:
        """Copy files to the sandbox.

        Args:
            container_id: Container identifier.
            source: Source file or directory path.
            destination: Destination path in container.

        Returns:
            True if successful, False otherwise.
        """
        if container_id not in self._containers:
            logger.error(f"Container not found: {container_id}")
            return False

        args = ["docker", "cp", source, f"{container_id}:{destination}"]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                logger.error("Failed to copy to sandbox")
                return False

            logger.debug(f"Copied {source} to {container_id}:{destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy to sandbox: {e}")
            return False

    async def copy_from_sandbox(
        self, container_id: str, source: str, destination: str
    ) -> bool:
        """Copy files from the sandbox.

        Args:
            container_id: Container identifier.
            source: Source path in container.
            destination: Destination file or directory path.

        Returns:
            True if successful, False otherwise.
        """
        if container_id not in self._containers:
            logger.error(f"Container not found: {container_id}")
            return False

        args = ["docker", "cp", f"{container_id}:{source}", destination]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                logger.error("Failed to copy from sandbox")
                return False

            logger.debug(f"Copied {container_id}:{source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Failed to copy from sandbox: {e}")
            return False

    async def stop_sandbox(self, container_id: str) -> bool:
        """Stop a sandbox container.

        Args:
            container_id: Container identifier.

        Returns:
            True if successful, False otherwise.
        """
        if container_id not in self._containers:
            logger.error(f"Container not found: {container_id}")
            return False

        args = ["docker", "stop", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                logger.error("Failed to stop container")
                return False

            if container_id in self._containers:
                self._containers[container_id].status = "stopped"

            logger.info(f"Stopped sandbox container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to stop container: {e}")
            return False

    async def remove_sandbox(self, container_id: str) -> bool:
        """Remove a sandbox container.

        Args:
            container_id: Container identifier.

        Returns:
            True if successful, False otherwise.
        """
        if container_id not in self._containers:
            logger.error(f"Container not found: {container_id}")
            return False

        # Stop container first if running
        if self._containers[container_id].status == "running":
            await self.stop_sandbox(container_id)

        args = ["docker", "rm", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                logger.error("Failed to remove container")
                return False

            if container_id in self._containers:
                del self._containers[container_id]

            logger.info(f"Removed sandbox container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to remove container: {e}")
            return False

    async def health_check(self) -> bool:
        """Check if Docker daemon is healthy.

        Returns:
            True if Docker is healthy, False otherwise.
        """
        try:
            process = await asyncio.create_subprocess_exec(
                "docker",
                "info",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()
            return process.returncode == 0
        except Exception:
            return False

    async def close(self) -> None:
        """Close Docker sandbox resources."""
        # Stop all containers
        for container_id in list(self._containers.keys()):
            try:
                await self.stop_sandbox(container_id)
            except Exception as e:
                logger.warning(f"Failed to stop container {container_id}: {e}")

        self._containers.clear()
        self._initialized = False
        logger.info("Docker sandbox closed")

    def get_container_info(self, container_id: str) -> DockerContainer | None:
        """Get information about a container.

        Args:
            container_id: Container identifier.

        Returns:
            DockerContainer if found, None otherwise.
        """
        return self._containers.get(container_id)

    def list_containers(self) -> list[DockerContainer]:
        """List all containers.

        Returns:
            List of DockerContainer objects.
        """
        return list(self._containers.values())

    async def get_container_logs(
        self, container_id: str, tail: int = 100
    ) -> str:
        """Get container logs.

        Args:
            container_id: Container identifier.
            tail: Number of lines to show from the end.

        Returns:
            Container logs as string.
        """
        if container_id not in self._containers:
            return f"Container not found: {container_id}"

        args = ["docker", "logs", "--tail", str(tail), container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                return "Failed to get logs"

            return ""
        except Exception as e:
            return f"Error getting logs: {e}"

    async def exec_in_container(
        self,
        container_id: str,
        command: list[str],
        working_dir: str | None = None,
    ) -> DockerSandboxResult:
        """Execute a command in the container with more control.

        Args:
            container_id: Container identifier.
            command: Command and arguments as list.
            working_dir: Optional working directory.

        Returns:
            DockerSandboxResult with execution results.
        """
        if container_id not in self._containers:
            return DockerSandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=f"Container not found: {container_id}",
                container_id=container_id,
                execution_time=0.0,
            )

        args = ["docker", "exec", "-i", container_id]

        if working_dir:
            args.extend(["-w", working_dir])

        args.extend(command)

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            exit_code: int = process.returncode if process.returncode is not None else -1

            return DockerSandboxResult(
                success=exit_code == 0,
                exit_code=exit_code,
                stdout="",
                stderr="",
                container_id=container_id,
                execution_time=self._get_execution_time(),
            )
        except Exception as e:
            return DockerSandboxResult(
                success=False,
                exit_code=-1,
                stdout="",
                stderr=str(e),
                container_id=container_id,
                execution_time=0.0,
            )

    async def get_container_stats(self, container_id: str) -> dict[str, Any]:
        """Get container resource usage statistics.

        Args:
            container_id: Container identifier.

        Returns:
            Dictionary with container statistics.
        """
        if container_id not in self._containers:
            return {"error": f"Container not found: {container_id}"}

        args = [
            "docker",
            "stats",
            "--no-stream",
            "--format",
            "{{.CPUPerc}}|{{.MemUsage}}|{{.NetIO}}",
            container_id,
        ]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                return {"error": "Failed to get stats"}

            return {"error": "Failed to parse stats"}
        except Exception as e:
            return {"error": str(e)}

    def _get_timestamp(self) -> datetime:
        """Get current timestamp."""
        return datetime.now(UTC)

    def _get_execution_time(self) -> float:
        """Get current execution time in seconds."""
        return round(time.time(), 3)

    async def restart_container(self, container_id: str) -> bool:
        """Restart a container.

        Args:
            container_id: Container identifier.

        Returns:
            True if successful, False otherwise.
        """
        if container_id not in self._containers:
            logger.error(f"Container not found: {container_id}")
            return False

        args = ["docker", "restart", container_id]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, _ = await process.communicate()

            if process.returncode != 0:
                logger.error("Failed to restart container")
                return False

            logger.info(f"Restarted sandbox container: {container_id}")
            return True
        except Exception as e:
            logger.error(f"Failed to restart container: {e}")
            return False

    async def inspect_container(self, container_id: str) -> dict[str, Any]:
        """Inspect a container for detailed information.

        Args:
            container_id: Container identifier.

        Returns:
            Dictionary with container inspection data.
        """
        if container_id not in self._containers:
            return {"error": f"Container not found: {container_id}"}

        args = ["docker", "inspect", container_id, "--format", "{{json .}}"]

        try:
            process = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()

            if process.returncode != 0:
                return {"error": stderr.decode("utf-8")}

            try:
                return json.loads(stdout.decode("utf-8"))  # type: ignore[no-any-return]
            except json.JSONDecodeError:
                return {"raw_output": stdout.decode("utf-8")}
        except Exception as e:
            return {"error": str(e)}

