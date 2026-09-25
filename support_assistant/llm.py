"""Optional real-LLM client (only used when MOCK_LLM=0).

The graded default never imports anything from here at call time: with MOCK_LLM unset
or "1", every node answers from deterministic code and no LLM request is made.
With MOCK_LLM=0 this calls Groq's free tier (OpenAI-compatible API), reading the key from
the GROQ_API_KEY environment variable (never hard-coded or committed).
"""

import json
import os
import re

import requests
from pydantic import ValidationError

from config import GROQ_MODEL, GROQ_URL, MAX_SCHEMA_RETRIES
from prompts import CORRECTIVE_INSTRUCTION
from schemas import AskResponse


def chat(messages: list[dict], temperature: float = 0.0) -> str:
    """Send one chat-completion request and return the reply text."""
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise RuntimeError("MOCK_LLM=0 requires the GROQ_API_KEY environment variable")
    response = requests.post(
        GROQ_URL,
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": GROQ_MODEL, "messages": messages, "temperature": temperature},
        timeout=30,
    )
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"].strip()


def _extract_json(text: str) -> str:
    """Strip markdown fences / surrounding prose that models sometimes add."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.M).strip()
    match = re.search(r"\{.*\}", text, flags=re.S)
    return match.group(0) if match else text


def generate_validated(prompt: str) -> AskResponse:
    """Ask the LLM for JSON and validate it against AskResponse.

    If validation fails, retry up to MAX_SCHEMA_RETRIES more times, each time adding a
    corrective instruction that quotes the validation error. After the last failure,
    return a clearly marked error response instead of raising.
    """
    messages = [{"role": "user", "content": prompt}]
    last_error = None
    for attempt in range(1 + MAX_SCHEMA_RETRIES):
        raw = chat(messages)
        try:
            return AskResponse.model_validate(json.loads(_extract_json(raw)))
        except (json.JSONDecodeError, ValidationError) as err:
            last_error = err
            messages += [
                {"role": "assistant", "content": raw},
                {"role": "user", "content": CORRECTIVE_INSTRUCTION.format(error=str(err)[:500])},
            ]
    return AskResponse(
        answer=f"[ERROR] The language model did not return valid JSON after "
               f"{1 + MAX_SCHEMA_RETRIES} attempts: {str(last_error)[:200]}",
        sources=[],
        confidence=0.0,
    )
