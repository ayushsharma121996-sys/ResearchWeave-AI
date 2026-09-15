"""
Thin wrapper supporting 100% Free Llama (via Groq Cloud API or Local Ollama),
Google Gemini API, and Anthropic SDK with JSON-mode helpers.

Centralizing this means:
  - one place to add retry-on-malformed-JSON logic
  - automatic rate-limit backoff handling
  - seamless switching between Llama (Groq / Ollama), Gemini, and Anthropic
  - easy request/response logging
"""
import json
import os
import re
import time
from typing import Any

import requests
from dotenv import load_dotenv

from app.config import (
    GROQ_API_KEY,
    OLLAMA_HOST,
    GEMINI_API_KEY,
    ANTHROPIC_API_KEY,
    MODEL_QA,
)

load_dotenv()


def _strip_json_fences(text: str) -> str:
    text = text.strip()
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    return text.strip()


def get_api_key() -> tuple[str, str]:
    """Return tuple of (provider, api_key_or_endpoint).
    Order of preference (Strictly 100% Free Llama):
    1. GROQ_API_KEY (Groq Cloud API for Llama 3.3 70B & 3.1 8B)
    2. Explicit LLM_PROVIDER ('ollama', 'groq', 'gemini', 'anthropic')
    3. Default to Local Ollama (http://localhost:11434) - 100% free, zero API keys
    """
    explicit_provider = os.environ.get("LLM_PROVIDER", "").lower()

    if explicit_provider == "groq":
        groq_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
        if groq_key:
            return ("groq", groq_key)
    elif explicit_provider == "gemini":
        gemini_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY") or GEMINI_API_KEY
        if gemini_key:
            return ("gemini", gemini_key)
    elif explicit_provider == "anthropic":
        anthropic_key = os.environ.get("ANTHROPIC_API_KEY") or ANTHROPIC_API_KEY
        if anthropic_key:
            return ("anthropic", anthropic_key)

    # Default to Groq Free Cloud API if key is present
    groq_key = os.environ.get("GROQ_API_KEY") or GROQ_API_KEY
    if groq_key:
        return ("groq", groq_key)

    # On Vercel / AWS Lambda, if no key is set, local Ollama cannot be reached
    if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
        raise RuntimeError(
            "GROQ_API_KEY is not set in Vercel Environment Variables. "
            "Please go to your Vercel Project Settings -> Environment Variables, "
            "add GROQ_API_KEY, and redeploy."
        )

    # Default to 100% Free Local Ollama Llama for local desktop use
    ollama_host = os.environ.get("OLLAMA_HOST") or OLLAMA_HOST or "http://localhost:11434"
    return ("ollama", ollama_host)



def call_llm(system: str, user: str, model: str = MODEL_QA, max_tokens: int = 1500, retries: int = 5) -> str:
    provider, credential = get_api_key()

    if provider == "groq":
        groq_model = "groq/compound" if ("gemini" in model.lower() or "claude" in model.lower() or "llama" in model.lower() or "flash" in model.lower()) else model
        headers = {
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": groq_model,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            "max_tokens": max_tokens,
        }
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    last_error = RuntimeError(f"Rate limit (429) hit on Groq: {resp.text}")
                    wait_time = 6.0 + (attempt * 4)
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                time.sleep(0.3)  # Pacing to respect Groq rate limits
                return data["choices"][0]["message"]["content"]
            except Exception as e:
                last_error = e
                if attempt == retries:
                    raise e
                time.sleep(2.0)
        raise RuntimeError(f"Failed to obtain response from Groq: {last_error}")

    elif provider == "ollama":
        ollama_model = "llama3.1" if ("gemini" in model.lower() or "claude" in model.lower() or "versatile" in model.lower()) else model
        url = f"{credential.rstrip('/')}/api/generate"
        payload = {
            "model": ollama_model,
            "system": system,
            "prompt": user,
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                return data.get("response", "")
            except Exception as e:
                if attempt == retries:
                    raise e
                time.sleep(2.0)
        raise RuntimeError(f"Failed to obtain response from Ollama at {credential}.")

    elif provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=credential)
        gemini_model = "gemini-3.6-flash" if ("gemini" in model.lower() or "claude" in model.lower() or "llama" in model.lower()) else model
        config = types.GenerateContentConfig(
            system_instruction=system,
            max_output_tokens=max_tokens,
        )

        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(
                    model=gemini_model,
                    contents=user,
                    config=config,
                )
                return resp.text
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    wait_time = 12.0 + (attempt * 4)
                    match = re.search(r"retry in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                    if match:
                        wait_time = float(match.group(1)) + 1.5
                    time.sleep(wait_time)
                    continue
                if attempt == retries:
                    raise e
                time.sleep(2.0)
        raise RuntimeError("Failed to obtain LLM response after retries.")

    else:
        import anthropic
        client = anthropic.Anthropic(api_key=credential)
        resp = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=system,
            messages=[{"role": "user", "content": user}],
        )
        return "".join(block.text for block in resp.content if block.type == "text")


def call_llm_json(system: str, user: str, model: str = MODEL_QA, max_tokens: int = 1500,
                   retries: int = 5) -> dict[str, Any]:
    provider, credential = get_api_key()

    if provider == "groq":
        groq_model = "groq/compound" if ("gemini" in model.lower() or "claude" in model.lower() or "llama" in model.lower() or "flash" in model.lower()) else model
        headers = {
            "Authorization": f"Bearer {credential}",
            "Content-Type": "application/json",
        }
        strict_system = system + "\n\nYou must respond with ONLY a valid JSON object. No markdown code fences, no preamble."
        payload = {
            "model": groq_model,
            "messages": [
                {"role": "system", "content": strict_system},
                {"role": "user", "content": user},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": max_tokens,
        }
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
                if resp.status_code == 429:
                    last_error = RuntimeError(f"Rate limit (429) hit on Groq: {resp.text}")
                    wait_time = 6.0 + (attempt * 4)
                    time.sleep(wait_time)
                    continue
                resp.raise_for_status()
                data = resp.json()
                raw_text = data["choices"][0]["message"]["content"]
                cleaned = _strip_json_fences(raw_text)
                time.sleep(0.3)  # Pacing to respect Groq rate limits
                return json.loads(cleaned)
            except Exception as e:
                last_error = e
                if attempt == retries:
                    raise ValueError(f"Groq LLM failed to return valid JSON after retries: {last_error}")
                time.sleep(1.0)
        raise ValueError(f"Groq LLM failed to return valid JSON after retries: {last_error}")

    elif provider == "ollama":
        ollama_model = "llama3.1" if ("gemini" in model.lower() or "claude" in model.lower() or "versatile" in model.lower()) else model
        url = f"{credential.rstrip('/')}/api/generate"
        payload = {
            "model": ollama_model,
            "system": system + "\n\nYou must respond with ONLY a valid JSON object.",
            "prompt": user,
            "format": "json",
            "stream": False,
            "options": {"num_predict": max_tokens},
        }
        last_error = None
        for attempt in range(retries + 1):
            try:
                resp = requests.post(url, json=payload, timeout=120)
                resp.raise_for_status()
                data = resp.json()
                cleaned = _strip_json_fences(data.get("response", ""))
                return json.loads(cleaned)
            except Exception as e:
                last_error = e
                if attempt == retries:
                    raise ValueError(f"Ollama LLM failed to return valid JSON after retries: {last_error}")
                time.sleep(1.0)
        raise ValueError(f"Ollama LLM failed to return valid JSON after retries: {last_error}")

    elif provider == "gemini":
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=credential)
        gemini_model = "gemini-3.6-flash" if ("gemini" in model.lower() or "claude" in model.lower() or "llama" in model.lower()) else model
        config = types.GenerateContentConfig(
            system_instruction=system + "\n\nYou must respond with ONLY a valid JSON object. No markdown fencing, no preamble.",
            response_mime_type="application/json",
            max_output_tokens=max_tokens,
        )

        last_error = None
        prompt = user
        for attempt in range(retries + 1):
            try:
                resp = client.models.generate_content(
                    model=gemini_model,
                    contents=prompt,
                    config=config,
                )
                cleaned = _strip_json_fences(resp.text)
                return json.loads(cleaned)
            except Exception as e:
                last_error = e
                err_str = str(e)
                if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "Quota exceeded" in err_str:
                    wait_time = 12.0 + (attempt * 4)
                    match = re.search(r"retry in (\d+\.?\d*)s", err_str, re.IGNORECASE)
                    if match:
                        wait_time = float(match.group(1)) + 1.5
                    time.sleep(wait_time)
                    continue
                prompt = user + f"\n\nPrevious response failed JSON parsing ({e}). Return ONLY valid JSON."
                time.sleep(1.0)
        raise ValueError(f"Gemini LLM failed to return valid JSON after retries: {last_error}")

    else:
        strict_system = (
            system
            + "\n\nYou must respond with ONLY a valid JSON object. No preamble, "
            "no markdown code fences, no explanation before or after."
        )
        last_error = None
        for attempt in range(retries + 1):
            raw = call_llm(strict_system, user, model=model, max_tokens=max_tokens)
            cleaned = _strip_json_fences(raw)
            try:
                return json.loads(cleaned)
            except json.JSONDecodeError as e:
                last_error = e
                user = (
                    user
                    + f"\n\nYour previous response was not valid JSON ({e}). "
                    "Return ONLY the corrected JSON object, nothing else."
                )
        raise ValueError(f"LLM failed to return valid JSON after {retries + 1} attempts: {last_error}")




