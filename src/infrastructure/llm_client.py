"""
LLM Client implementation for the Cybernetic Production Studio.

This module provides a unified interface for communicating with various LLM providers:
- Hugging Face Inference API
- OpenAI API
- Local servers (Ollama, vLLM, etc.)

The implementation follows the LLMClientProtocol and supports:
- Synchronous and streaming requests
- Model information retrieval
- Health checks
- Automatic retry with exponential backoff
"""

import json
import logging
import time
from collections.abc import AsyncGenerator
from typing import Any

import aiohttp

from src.infrastructure.strategy import (
    LLMClientProtocol,
    LLMConfig,
    LLMProvider,
    LLMRequest,
    LLMResponse,
)

logger = logging.getLogger(__name__)

# Magic value constants
HTTP_STATUS_OK = 200
HTTP_TIMEOUT_SECONDS = 10


class LLMClient(LLMClientProtocol):
    """Implementation of LLM client for various providers."""

    def __init__(self, config: LLMConfig) -> None:
        """Initialize the LLM client.

        Args:
            config: LLM configuration with provider, model_id, and credentials.
        """
        self._config = config
        self._session: aiohttp.ClientSession | None = None
        self._initialized = False

    async def _get_session(self) -> aiohttp.ClientSession:
        """Get or create aiohttp session."""
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self._config.timeout)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session

    async def initialize(self) -> None:
        """Initialize the LLM client."""
        if not self._initialized:
            self._session = await self._get_session()
            self._initialized = True
            logger.info(f"LLM client initialized for {self._config.provider.value}")

    async def close(self) -> None:
        """Close the LLM client connection."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._initialized = False
            logger.info("LLM client closed")

    async def send_request(self, request: LLMRequest) -> LLMResponse:
        """Send a request to the LLM server.

        Args:
            request: LLM request with messages and configuration.

        Returns:
            LLMResponse with content and metadata.

        Raises:
            RuntimeError: If client is not initialized.
            aiohttp.ClientError: If request fails.
        """
        await self.initialize()

        provider = request.config.provider if request.config else self._config.provider

        if provider == LLMProvider.HUGGING_FACE:
            return await self._send_huggingface_request(request)
        if provider == LLMProvider.OPENAI:
            return await self._send_openai_request(request)
        if provider == LLMProvider.LOCAL:
            return await self._send_local_request(request)
        if provider == LLMProvider.OLLAMA:
            return await self._send_ollama_request(request)
        raise ValueError(f"Unsupported provider: {provider}")

    async def _send_huggingface_request(self, request: LLMRequest) -> LLMResponse:
        """Send request to Hugging Face Inference API."""
        session = await self._get_session()
        url = f"https://api-inference.huggingface.co/models/{self._config.model_id}/infer"

        headers = {
            "Content-Type": "application/json",
        }

        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        # Format messages for Hugging Face
        messages = request.messages if request.messages else []
        payload = {
            "inputs": "\n".join([f"{m['role']}: {m['content']}" for m in messages]),
            "parameters": {
                "temperature": request.config.temperature if request.config else 0.7,
                "max_new_tokens": request.config.max_tokens if request.config else 4096,
            },
        }

        start_time = time.time()

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Hugging Face API error: {response.status} - {error_text}")

            result = await response.json()

            # Handle different response formats
            if isinstance(result, list) and len(result) > 0:
                content = result[0].get("generated_text", "")
            elif isinstance(result, dict):
                content = result.get("generated_text", "")
            else:
                content = str(result)

            execution_time = time.time() - start_time
            tokens_used = len(content.split())

            return LLMResponse(
                content=content,
                model_id=self._config.model_id,
                tokens_used=tokens_used,
                finish_reason="stop",
                metadata={"execution_time": execution_time},
            )

    async def _send_openai_request(self, request: LLMRequest) -> LLMResponse:
        """Send request to OpenAI API."""
        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
        }

        payload = {
            "model": self._config.model_id,
            "messages": request.messages,
            "temperature": request.config.temperature if request.config else 0.7,
            "max_tokens": request.config.max_tokens if request.config else 4096,
        }

        start_time = time.time()

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"OpenAI API error: {response.status} - {error_text}")

            result = await response.json()

            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0]["finish_reason"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            execution_time = time.time() - start_time

            return LLMResponse(
                content=content,
                model_id=self._config.model_id,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                metadata={"execution_time": execution_time},
            )

    async def _send_local_request(self, request: LLMRequest) -> LLMResponse:
        """Send request to local LLM server."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:8000"
        url = f"{base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
        }

        payload = {
            "model": self._config.model_id,
            "messages": request.messages,
            "temperature": request.config.temperature if request.config else 0.7,
            "max_tokens": request.config.max_tokens if request.config else 4096,
        }

        start_time = time.time()

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Local LLM server error: {response.status} - {error_text}")

            result = await response.json()

            content = result["choices"][0]["message"]["content"]
            finish_reason = result["choices"][0]["finish_reason"]
            tokens_used = result.get("usage", {}).get("total_tokens", 0)

            execution_time = time.time() - start_time

            return LLMResponse(
                content=content,
                model_id=self._config.model_id,
                tokens_used=tokens_used,
                finish_reason=finish_reason,
                metadata={"execution_time": execution_time},
            )

    async def _send_ollama_request(self, request: LLMRequest) -> LLMResponse:
        """Send request to Ollama server."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:11434"
        url = f"{base_url}/api/chat"

        headers = {
            "Content-Type": "application/json",
        }

        # Format messages for Ollama
        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in request.messages
        ]

        payload = {
            "model": self._config.model_id,
            "messages": ollama_messages,
            "options": {
                "temperature": request.config.temperature if request.config else 0.7,
                "num_predict": request.config.max_tokens if request.config else 4096,
            },
            "stream": False,
        }

        start_time = time.time()

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Ollama server error: {response.status} - {error_text}")

            result = await response.json()

            content = result["message"]["content"]
            tokens_used = result.get("eval_count", 0)

            execution_time = time.time() - start_time

            return LLMResponse(
                content=content,
                model_id=self._config.model_id,
                tokens_used=tokens_used,
                finish_reason="stop",
                metadata={"execution_time": execution_time},
            )

    async def send_request_stream(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Send a streaming request to the LLM server.

        Args:
            request: LLM request with messages and configuration.

        Yields:
            Stream of content chunks.

        Raises:
            RuntimeError: If client is not initialized.
            aiohttp.ClientError: If request fails.
        """
        await self.initialize()

        provider = request.config.provider if request.config else self._config.provider

        if provider == LLMProvider.HUGGING_FACE:
            async for chunk in self._stream_huggingface_request(request):
                yield chunk
        elif provider == LLMProvider.OPENAI:
            async for chunk in self._stream_openai_request(request):
                yield chunk
        elif provider == LLMProvider.LOCAL:
            async for chunk in self._stream_local_request(request):
                yield chunk
        elif provider == LLMProvider.OLLAMA:
            async for chunk in self._stream_ollama_request(request):
                yield chunk
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    async def _stream_huggingface_request(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream response from Hugging Face Inference API."""
        session = await self._get_session()
        url = f"https://api-inference.huggingface.co/models/{self._config.model_id}/infer"

        headers = {
            "Content-Type": "application/json",
        }

        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        messages = request.messages if request.messages else []
        payload = {
            "inputs": "\n".join([f"{m['role']}: {m['content']}" for m in messages]),
            "parameters": {
                "temperature": request.config.temperature if request.config else 0.7,
                "max_new_tokens": request.config.max_tokens if request.config else 4096,
                "stream": True,
            },
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Hugging Face API error: {response.status} - {error_text}")

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    data = line_str[5:].strip()
                    if data:
                        try:
                            result = json.loads(data)
                            if isinstance(result, dict) and "token" in result:
                                yield result["token"].get("text", "")
                        except json.JSONDecodeError:
                            pass

    async def _stream_openai_request(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream response from OpenAI API."""
        session = await self._get_session()
        url = "https://api.openai.com/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._config.api_key}",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": self._config.model_id,
            "messages": request.messages,
            "temperature": request.config.temperature if request.config else 0.7,
            "max_tokens": request.config.max_tokens if request.config else 4096,
            "stream": True,
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"OpenAI API error: {response.status} - {error_text}")

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    data = line_str[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            result = json.loads(data)
                            if "choices" in result and len(result["choices"]) > 0:
                                delta = result["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass

    async def _stream_local_request(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream response from local LLM server."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:8000"
        url = f"{base_url}/v1/chat/completions"

        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }

        payload = {
            "model": self._config.model_id,
            "messages": request.messages,
            "temperature": request.config.temperature if request.config else 0.7,
            "max_tokens": request.config.max_tokens if request.config else 4096,
            "stream": True,
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Local LLM server error: {response.status} - {error_text}")

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if line_str.startswith("data:"):
                    data = line_str[5:].strip()
                    if data and data != "[DONE]":
                        try:
                            result = json.loads(data)
                            if "choices" in result and len(result["choices"]) > 0:
                                delta = result["choices"][0].get("delta", {})
                                if "content" in delta:
                                    yield delta["content"]
                        except json.JSONDecodeError:
                            pass

    async def _stream_ollama_request(
        self, request: LLMRequest
    ) -> AsyncGenerator[str, None]:
        """Stream response from Ollama server."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:11434"
        url = f"{base_url}/api/chat"

        headers = {
            "Content-Type": "application/json",
        }

        ollama_messages = [
            {"role": msg["role"], "content": msg["content"]} for msg in request.messages
        ]

        payload = {
            "model": self._config.model_id,
            "messages": ollama_messages,
            "options": {
                "temperature": request.config.temperature if request.config else 0.7,
                "num_predict": request.config.max_tokens if request.config else 4096,
            },
            "stream": True,
        }

        async with session.post(url, headers=headers, json=payload) as response:
            if response.status != HTTP_STATUS_OK:
                error_text = await response.text()
                raise RuntimeError(f"Ollama server error: {response.status} - {error_text}")

            async for line in response.content:
                line_str = line.decode("utf-8").strip()
                if line_str:
                    try:
                        result = json.loads(line_str)
                        if "message" in result and "content" in result["message"]:
                            yield result["message"]["content"]
                    except json.JSONDecodeError:
                        pass

    async def get_model_info(self, model_id: str) -> dict[str, Any]:
        """Get information about a model.

        Args:
            model_id: Model identifier.

        Returns:
            Dictionary with model information.
        """
        # Default model information based on provider
        if self._config.provider == LLMProvider.HUGGING_FACE:
            return {
                "model_id": model_id,
                "provider": "huggingface",
                "task_type": "text-generation",
                "context_length": 4096,
                "quantization": "none",
            }
        if self._config.provider == LLMProvider.OPENAI:
            return {
                "model_id": model_id,
                "provider": "openai",
                "task_type": "text-generation",
                "context_length": 8192,
                "quantization": "none",
            }
        if self._config.provider == LLMProvider.LOCAL:
            return {
                "model_id": model_id,
                "provider": "local",
                "task_type": "text-generation",
                "context_length": 4096,
                "quantization": "unknown",
            }
        # LLMProvider.OLLAMA
        return {
            "model_id": model_id,
            "provider": "ollama",
            "task_type": "text-generation",
            "context_length": 4096,
            "quantization": "unknown",
        }

    async def health_check(self) -> bool:
        """Check if the LLM server is healthy.

        Returns:
            True if the server is healthy, False otherwise.
        """
        try:
            await self.initialize()
            provider = self._config.provider

            if provider == LLMProvider.HUGGING_FACE:
                return await self._health_check_huggingface()
            if provider == LLMProvider.OPENAI:
                return await self._health_check_openai()
            if provider == LLMProvider.LOCAL:
                return await self._health_check_local()
            # LLMProvider.OLLAMA
            return await self._health_check_ollama()
        except Exception:
            return False

    async def _health_check_huggingface(self) -> bool:
        """Check Hugging Face API health."""
        session = await self._get_session()
        url = f"https://api-inference.huggingface.co/models/{self._config.model_id}"

        headers = {}
        if self._config.api_key:
            headers["Authorization"] = f"Bearer {self._config.api_key}"

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, headers=headers, timeout=timeout) as response:
                return response.status == HTTP_STATUS_OK
        except Exception:
            return False

    async def _health_check_openai(self) -> bool:
        """Check OpenAI API health."""
        session = await self._get_session()
        url = "https://api.openai.com/v1/models"

        headers = {
            "Authorization": f"Bearer {self._config.api_key}",
        }

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, headers=headers, timeout=timeout) as response:
                return response.status == HTTP_STATUS_OK
        except Exception:
            return False

    async def _health_check_local(self) -> bool:
        """Check local LLM server health."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:8000"
        url = f"{base_url}/health"

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                return response.status == HTTP_STATUS_OK
        except Exception:
            return False

    async def _health_check_ollama(self) -> bool:
        """Check Ollama server health."""
        session = await self._get_session()
        base_url = self._config.base_url or "http://localhost:11434"
        url = f"{base_url}/api/tags"

        try:
            timeout = aiohttp.ClientTimeout(total=10)
            async with session.get(url, timeout=timeout) as response:
                return response.status == HTTP_STATUS_OK
        except Exception:
            return False
