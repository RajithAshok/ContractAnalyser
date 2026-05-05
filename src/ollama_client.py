"""
Thin wrapper around the Ollama HTTP API.
Handles model selection, JSON mode, and structured output prompting.
"""
from __future__ import annotations
import json
import re
import httpx
from typing import Optional


OLLAMA_BASE = "http://localhost:11434"

# Ordered preference list — first available model wins
PREFERRED_MODELS = ["llama3.2", "llama3.1", "mistral", "llama2", "phi3"]


def get_available_model() -> str:
    """Return the first preferred model that Ollama has downloaded."""
    try:
        r = httpx.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        r.raise_for_status()
        downloaded = {m["name"].split(":")[0] for m in r.json().get("models", [])}
        for m in PREFERRED_MODELS:
            if m in downloaded:
                return m
        # Fall back to whatever is available
        if downloaded:
            return next(iter(downloaded))
        raise RuntimeError("No models found. Run: ollama pull llama3.2")
    except httpx.ConnectError:
        raise RuntimeError(
            "Cannot connect to Ollama. Make sure it is running: ollama serve"
        )


def chat(
    prompt: str,
    system: Optional[str] = None,
    model: Optional[str] = None,
    expect_json: bool = False,
    temperature: float = 0.1,
) -> str:
    """
    Send a prompt to Ollama and return the response text.

    If expect_json=True, wraps the prompt with instructions to return
    only valid JSON and strips any markdown fences from the output.
    """
    if model is None:
        model = get_available_model()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})

    user_content = prompt
    if expect_json:
        user_content = (
            prompt
            + "\n\nIMPORTANT: Respond ONLY with valid JSON. No explanation, no markdown "
            "code fences, no preamble. Start your response with { or [."
        )
    messages.append({"role": "user", "content": user_content})

    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"temperature": temperature},
    }

    try:
        r = httpx.post(
            f"{OLLAMA_BASE}/api/chat",
            json=payload,
            timeout=120,
        )
        r.raise_for_status()
        text = r.json()["message"]["content"].strip()
    except httpx.ConnectError:
        raise RuntimeError("Lost connection to Ollama. Is it still running?")

    if expect_json:
        text = _strip_json_fences(text)

    return text


def _strip_json_fences(text: str) -> str:
    """Remove ```json ... ``` or ``` ... ``` wrappers if the model added them."""
    text = text.strip()
    # Remove opening fence
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    # Remove closing fence
    text = re.sub(r"\n?```\s*$", "", text)
    return text.strip()


def parse_json_response(text: str) -> dict | list:
    """
    Parse a JSON response from the model. Falls back to extracting the
    first { ... } or [ ... ] block if the model added surrounding text.
    """
    text = _strip_json_fences(text)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find a JSON object or array in the text
        match = re.search(r"(\{[\s\S]*\}|\[[\s\S]*\])", text)
        if match:
            return json.loads(match.group(1))
        raise ValueError(f"Could not parse JSON from model output:\n{text[:500]}")
