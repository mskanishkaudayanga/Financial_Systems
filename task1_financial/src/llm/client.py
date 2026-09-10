"""
LLM Client Wrapper.

Provides an abstraction interface for invoking LLM services via REST API
with structured JSON generation, configurable temperature, and error handling.
"""

import json
import logging
from typing import Any, Dict, Optional
import requests

from src.config import config

logger = logging.getLogger(__name__)


class LLMClientError(Exception):
    """Base exception for LLM client communication and execution failures."""
    pass


class LLMClient:
    """
    OpenAI-compatible REST API client wrapper.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model_name: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: Optional[float] = None,
        timeout: Optional[int] = None,
    ):
        self.api_key = api_key if api_key is not None else config.LLM_API_KEY
        self.model_name = model_name if model_name is not None else config.LLM_MODEL_NAME
        self.base_url = (base_url if base_url is not None else config.LLM_BASE_URL).rstrip("/")
        self.temperature = temperature if temperature is not None else config.LLM_TEMPERATURE
        self.timeout = timeout if timeout is not None else config.LLM_TIMEOUT

        self.session = requests.Session()

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None
    ) -> str:
        """
        Send a chat completion request to the LLM REST API.

        Args:
            prompt: User prompt text.
            system_prompt: Optional system prompt instructions.
            temperature: Optional override for sampling temperature.

        Returns:
            str: Raw LLM text response.

        Raises:
            LLMClientError: If request fails due to missing key, network error, or HTTP status error.
        """
        if not self.api_key:
            raise LLMClientError("LLM API key is not configured in environment variables.")

        temp = temperature if temperature is not None else self.temperature

        url = f"{self.base_url}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        payload = {
            "model": self.model_name,
            "messages": messages,
            "temperature": temp,
        }

        try:
            response = self.session.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()
            data = response.json()

            choices = data.get("choices", [])
            if not choices or not isinstance(choices, list):
                raise LLMClientError("Invalid response structure from LLM API: missing 'choices'.")

            content = choices[0].get("message", {}).get("content", "")
            if not content or not isinstance(content, str):
                raise LLMClientError("Empty response content received from LLM API.")

            return content.strip()

        except requests.exceptions.Timeout as exc:
            raise LLMClientError(f"LLM API request timed out after {self.timeout} seconds.") from exc
        except requests.exceptions.RequestException as exc:
            raise LLMClientError(f"LLM API HTTP communication error: {exc}") from exc
        except (json.JSONDecodeError, KeyError, IndexError) as exc:
            raise LLMClientError(f"Failed to parse LLM API response payload: {exc}") from exc
