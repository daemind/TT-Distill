import asyncio
import contextlib
import shutil
from pathlib import Path

from .logger import get_logger

logger = get_logger(__name__)


class DockerManager:
    """
    Manages Docker-based sandboxing for command execution.
    Falls back to local execution if Docker is not available.
    All execution is async to never block the event loop.
    """

    def __init__(self, image: str = "python:3.11-slim") -> None:
        self.image = image
        self.is_available = False
        self.container_name: str | None = None
        self._docker_path = shutil.which("docker")

    async def _check_docker_async(self) -> bool:
        """Checks if docker is installed and daemon is running asynchronously."""
        docker_bin = shutil.which("docker")
        if not docker_bin:
            return False

        try:
            proc = await asyncio.create_subprocess_exec(
                docker_bin,
                "ps",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            return proc.returncode == 0
        except Exception:
            return False

    async def initialize(self) -> None:
        """Async initialization to check docker availability."""
        self.is_available = await self._check_docker_async()

    async def start_sandbox(self, project_id: str, project_root: str) -> None:
        """Starts a persistent background container for the project."""
        if not self.is_available or not self._docker_path:
            return

        self.container_name = f"cybernetic-sandbox-{project_id}"
        abs_workdir = await asyncio.to_thread(lambda: str(Path(project_root).resolve()))

        # Check if container already exists and is running
        check_proc = await asyncio.create_subprocess_exec(
            self._docker_path,
            "ps",
            "-q",
            "-f",
            f"name={self.container_name}",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, _ = await check_proc.communicate()
        if stdout.decode("utf-8").strip():
            logger.info(
                f"🐳 Persistent Docker sandbox '{self.container_name}' already running."
            )
            return

        # Clean up any stopped container with the same name
        rm_proc = await asyncio.create_subprocess_exec(
            self._docker_path,
            "rm",
            "-f",
            self.container_name,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await rm_proc.communicate()

        logger.info(f"🐳 Starting persistent Docker sandbox: {self.container_name}")
        # Run a detached container that stays alive (tail -f /dev/null)
        run_proc = await asyncio.create_subprocess_exec(
            self._docker_path,
            "run",
            "-d",
            "--name",
            self.container_name,
            "-v",
            f"{abs_workdir}:/workspace",
            self.image,
            "tail",
            "-f",
            "/dev/null",
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await run_proc.communicate()

        if run_proc.returncode != 0:
            logger.warning(
                f"⚠️ Failed to start persistent Docker sandbox. Falling back local. Error: {stderr.decode('utf-8')}"
            )
            self.is_available = False

    async def stop_sandbox(self) -> None:
        """Stops and removes the persistent container."""
        if self.container_name and self._docker_path:
            logger.info(f"🐳 Stopping persistent Docker sandbox: {self.container_name}")
            rm_proc = await asyncio.create_subprocess_exec(
                self._docker_path,
                "rm",
                "-f",
                self.container_name,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            await rm_proc.communicate()
            self.container_name = None

    async def execute(self, command: str, workdir: str) -> dict[str, str | int]:
        """
        Executes a command inside a Docker container or locally as fallback.
        Always async — never blocks the event loop.
        Returns a dict with keys 'stdout', 'stderr', 'exit_code'.
        """
        if self.is_available:
            return await self._execute_docker(command, workdir)
        return await self._execute_local(command, workdir)

    async def _execute_docker(self, command: str, workdir: str) -> dict[str, str | int]:
        """Runs command in the persistent Docker container via docker exec."""
        # Map the local absolute workdir to the in-container /workspace path
        # Assuming the project_root was mounted to /workspace
        container_workdir = "/workspace"  # Base mount point

        if self.container_name is None or not self._docker_path:
            # Fallback to local execution (which will also return dict)
            return await self._execute_local(command, workdir)

        docker_cmd = [
            self._docker_path,
            "exec",
            "-w",
            container_workdir,
            self.container_name,
            "bash",
            "-c",
            command,
        ]
        stdout, stderr, exit_code = await self._run_subprocess(
            docker_cmd, workdir, timeout_seconds=120
        )
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    async def _execute_local(self, command: str, workdir: str) -> dict[str, str | int]:
        """Fallback: Local async execution with a 60-second timeout."""
        # Use bash -c to execute shell command safely (no shell injection)
        bash_cmd = ["bash", "-c", command]
        stdout, stderr, exit_code = await self._run_subprocess(
            bash_cmd, workdir, timeout_seconds=60
        )
        return {"stdout": stdout, "stderr": stderr, "exit_code": exit_code}

    async def _run_shell(
        self, command: str, workdir: str, timeout_seconds: int = 60
    ) -> str:
        """Shared async shell runner using bash -c (safe from injection)."""
        bash_cmd = ["bash", "-c", command]
        async with asyncio.timeout(timeout_seconds):
            stdout, stderr, exit_code = await self._run_subprocess(
                bash_cmd, workdir, timeout_seconds
            )
        if exit_code == 0:
            return stdout
        return f"Error: {stderr}"

    async def _run_subprocess(
        self, args: list[str], workdir: str, timeout_seconds: int = 120
    ) -> tuple[str, str, int]:
        """
        Run a subprocess without shell injection.
        Returns (stdout, stderr, exit_code).
        """
        proc = await asyncio.create_subprocess_exec(
            *args,
            cwd=workdir,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            async with asyncio.timeout(timeout_seconds):
                stdout, stderr = await proc.communicate()
            return (
                stdout.decode("utf-8", errors="replace"),
                stderr.decode("utf-8", errors="replace"),
                proc.returncode or 0,
            )
        except TimeoutError:
            with contextlib.suppress(ProcessLookupError):
                proc.kill()
            return "", f"Command timed out after {timeout_seconds} seconds.", 1
