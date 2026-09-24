"""
OpenRouter LLM Integration via LangChain
Provides ChatOpenRouter integration with graceful error handling.
"""

import logging
import sys
from typing import Optional, List, Any

from langchain_core.messages import BaseMessage
from langchain_openrouter import ChatOpenRouter

from backend.config import (
    OPENROUTER_API_KEY,
    OPENROUTER_MODEL,
    is_openrouter_configured,
)


logger = logging.getLogger(__name__)


def _extract_text(content: Any) -> str:
    """
    Extract clean text from LangChain response content.
    """

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts = []

        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))

            elif hasattr(part, "text"):
                parts.append(str(part.text))

            elif isinstance(part, str):
                parts.append(part)

            else:
                parts.append(str(part))

        return "\n".join(parts).strip()

    return str(content).strip()


def get_llm(
    model_name: Optional[str] = None,
    temperature: float = 0.7
) -> Optional[ChatOpenRouter]:
    """
    Create and return an OpenRouter LangChain model.

    Returns None when the OpenRouter API key
    is not configured.
    """

    if not is_openrouter_configured():
        return None

    chosen_model = model_name or OPENROUTER_MODEL

    try:
        llm = ChatOpenRouter(
            model=chosen_model,
            api_key=OPENROUTER_API_KEY,
            temperature=temperature,
            max_tokens=2048,
        )

        return llm

    except Exception as exc:
        exc_type = type(exc).__name__

        print(
            f"[ERROR] Failed to initialize OpenRouter "
            f"with model {chosen_model} "
            f"[{exc_type}]: {exc}",
            file=sys.stderr,
        )

        logger.error(
            "Failed to initialize OpenRouter model %s [%s]: %s",
            chosen_model,
            exc_type,
            exc,
        )

        return None


def invoke_llm_safely(
    messages: List[BaseMessage],
    model_name: Optional[str] = None
) -> tuple[str, bool]:
    """
    Safely invoke OpenRouter.

    Returns:
        (response_text, is_success)
    """

    if not is_openrouter_configured():
        return (
            "AI service is temporarily unavailable. "
            "Please check your OpenRouter API configuration.",
            False,
        )

    llm = get_llm(model_name)

    if llm is None:
        return (
            "AI service is temporarily unavailable. "
            "Please check your OpenRouter configuration.",
            False,
        )

    try:
        response = llm.invoke(messages)

        text = _extract_text(response.content)

        if not text:
            return (
                "The AI returned an empty response. "
                "Please try again.",
                False,
            )

        return text, True

    except Exception as exc:
        exc_type = type(exc).__name__
        err_msg = str(exc)

        print(
            f"[ERROR] OpenRouter API invocation failed "
            f"[{exc_type}]: {err_msg}",
            file=sys.stderr,
        )

        logger.error(
            "OpenRouter API invocation failed [%s]: %s",
            exc_type,
            err_msg,
        )

        error_lower = err_msg.lower()

        # Invalid API key / authentication
        if (
            "401" in err_msg
            or "unauthorized" in error_lower
            or "invalid api key" in error_lower
            or "authentication" in error_lower
        ):
            return (
                "Invalid OpenRouter API key. "
                "Please verify OPENROUTER_API_KEY in your .env file.",
                False,
            )

        # Rate limit
        if (
            "429" in err_msg
            or "rate limit" in error_lower
            or "too many requests" in error_lower
        ):
            return (
                "OpenRouter rate limit exceeded. "
                "Please wait a moment and try again.",
                False,
            )

        # Credits / quota
        if (
            "402" in err_msg
            or "credits" in error_lower
            or "quota" in error_lower
        ):
            return (
                "OpenRouter quota or credits are currently unavailable. "
                "Please check your OpenRouter account.",
                False,
            )

        # Model unavailable
        if (
            "model" in error_lower
            and (
                "not found" in error_lower
                or "unavailable" in error_lower
                or "no endpoints" in error_lower
            )
        ):
            return (
                "The selected OpenRouter model is currently unavailable. "
                "Please try again later.",
                False,
            )

        # Generic fallback
        return (
            "Something went wrong while generating the answer "
            "through OpenRouter. Please try again.",
            False,
        )