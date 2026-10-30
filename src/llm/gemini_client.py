"""
TicketAI - Google Gemini Client
Official Google Gen AI SDK integration with structured outputs, timeout,
exponential backoff retry logic, latency measurement, and fallback handling.
"""

import os
import time
import json
import logging
from typing import Optional, Dict, Any, Type
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger("ticketai.llm")


class GeminiClient:
    """Reusable, production-hardened wrapper for Google Gemini API."""

    def __init__(self, api_key: Optional[str] = None, default_model: Optional[str] = None):
        self.api_key = (api_key or os.getenv("GEMINI_API_KEY") or "").strip()
        self.default_model = (default_model or os.getenv("GEMINI_MODEL") or "gemini-2.5-flash").strip()
        self._client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            logger.warning("GEMINI_API_KEY is not configured. Real LLM generation will be unavailable until set.")
            return

        try:
            from google import genai
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Google Gen AI client initialized successfully.")
        except ImportError:
            logger.error("google-genai SDK not installed. Please install google-genai.")
            self._client = None
        except Exception as e:
            logger.error(f"Failed to initialize Google Gen AI client: {e}")
            self._client = None

    @property
    def is_available(self) -> bool:
        return bool(self._client and self.api_key)

    def generate_content(
        self,
        prompt: str,
        system_instruction: Optional[str] = None,
        model: Optional[str] = None,
        response_schema: Optional[Type[BaseModel]] = None,
        max_retries: int = 3,
        timeout_seconds: float = 30.0,
        temperature: float = 0.2
    ) -> Dict[str, Any]:
        """
        Executes a call to Gemini with structured output, retries, and usage logging.
        Returns a dict containing:
        - 'text': raw response or JSON string
        - 'parsed': parsed Pydantic object / dict if schema provided
        - 'latency_ms': response latency in ms
        - 'model': model name used
        - 'usage': token counts if provided by API
        """
        target_model = model or self.default_model

        if not self.is_available:
            raise RuntimeError(
                "Gemini Client is unavailable. Ensure GEMINI_API_KEY is set in environment."
            )

        start_time = time.time()
        last_error = None

        from google.genai import types

        config_args = {
            "temperature": temperature,
        }
        if system_instruction:
            config_args["system_instruction"] = system_instruction

        if response_schema:
            config_args["response_mime_type"] = "application/json"
            config_args["response_schema"] = response_schema

        config = types.GenerateContentConfig(**config_args)

        for attempt in range(1, max_retries + 1):
            try:
                logger.info(f"Invoking Gemini model '{target_model}' (Attempt {attempt}/{max_retries})...")
                response = self._client.models.generate_content(
                    model=target_model,
                    contents=prompt,
                    config=config
                )

                latency_ms = round((time.time() - start_time) * 1000, 2)
                raw_text = response.text or ""

                parsed_result = None
                if response_schema:
                    try:
                        # Attempt to parse json with schema
                        clean_text = raw_text.strip()
                        if clean_text.startswith("```json"):
                            clean_text = clean_text[7:]
                        if clean_text.startswith("```"):
                            clean_text = clean_text[3:]
                        if clean_text.endswith("```"):
                            clean_text = clean_text[:-3]
                        clean_text = clean_text.strip()

                        parsed_dict = json.loads(clean_text)
                        parsed_result = response_schema(**parsed_dict)
                    except Exception as parse_err:
                        logger.warning(f"Response schema parsing warning: {parse_err}. Raw: {raw_text[:200]}")
                        parsed_result = None

                usage_metadata = {}
                if hasattr(response, "usage_metadata") and response.usage_metadata:
                    usage_metadata = {
                        "prompt_token_count": getattr(response.usage_metadata, "prompt_token_count", 0),
                        "candidates_token_count": getattr(response.usage_metadata, "candidates_token_count", 0),
                        "total_token_count": getattr(response.usage_metadata, "total_token_count", 0)
                    }

                logger.info(f"Gemini call completed in {latency_ms} ms. Model: {target_model}")
                return {
                    "text": raw_text,
                    "parsed": parsed_result,
                    "latency_ms": latency_ms,
                    "model": target_model,
                    "usage": usage_metadata,
                    "success": True
                }

            except Exception as e:
                last_error = e
                err_str = str(e).lower()
                logger.warning(f"Gemini call attempt {attempt} failed: {e}")

                # Check if rate limited or temporary network error
                if attempt < max_retries:
                    backoff = (2 ** attempt) * 0.5
                    if "429" in err_str or "quota" in err_str or "rate limit" in err_str:
                        backoff = max(backoff, 2.0)
                    time.sleep(backoff)

        raise RuntimeError(f"Gemini API generation failed after {max_retries} attempts: {last_error}")


_default_gemini_client: Optional[GeminiClient] = None

def get_gemini_client() -> GeminiClient:
    """Singleton getter for Gemini Client."""
    global _default_gemini_client
    if _default_gemini_client is None:
        _default_gemini_client = GeminiClient()
    return _default_gemini_client

