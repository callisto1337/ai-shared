import json
import logging
import os
import re
from typing import Any

import httpx
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Lead Scanner Model API")

VLLM_HOST = os.getenv("VLLM_HOST", "http://vllm:8000")
DEFAULT_MODEL = os.getenv(
    "DEFAULT_MODEL",
    "Qwen/Qwen3-14B-AWQ",
)
VLLM_TIMEOUT = float(os.getenv("VLLM_TIMEOUT", "90"))


class GenerateRequest(BaseModel):
    prompt: str = Field(min_length=1)
    model: str | None = None
    temperature: float = 0
    format: str | None = "json"


class GenerateResponse(BaseModel):
    ok: bool
    data: dict[str, Any] | None = None
    raw: str
    error: str | None = None


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "niche_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "intent_score": {
            "type": "integer",
            "minimum": 0,
            "maximum": 100,
        },
        "description": {
            "type": "string",
        },
    },
    "required": [
        "niche_score",
        "intent_score",
        "description",
    ],
    "additionalProperties": False,
}


MODEL_ALIASES = {
    "qwen3:14b": DEFAULT_MODEL,
}


def resolve_model(model: str | None) -> str:
    if model is None:
        return DEFAULT_MODEL

    return MODEL_ALIASES.get(model, model)


def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()

    try:
        result = json.loads(raw)
    except json.JSONDecodeError:
        result = None

    if isinstance(result, dict):
        return result

    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if match is None:
        raise ValueError(
            f"No JSON found in model response: {raw}"
        )

    result = json.loads(match.group(0))

    if not isinstance(result, dict):
        raise ValueError("Model response is not a JSON object")

    return result


@app.get("/health")
async def health() -> dict[str, str]:
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(
                f"{VLLM_HOST}/v1/models"
            )
            response.raise_for_status()

        return {
            "status": "ok",
            "vllm": "available",
            "default_model": DEFAULT_MODEL,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"vLLM unavailable: {exc}",
        ) from exc


@app.post(
    "/generate",
    response_model=GenerateResponse,
)
async def generate(
    request: GenerateRequest,
) -> GenerateResponse:
    model = resolve_model(request.model)

    payload: dict[str, Any] = {
        "model": model,
        "messages": [
            {
                "role": "user",
                "content": request.prompt,
            }
        ],
        "stream": False,
        "temperature": request.temperature,
        "max_tokens": 128,
        "chat_template_kwargs": {
            "enable_thinking": False,
        },
    }

    if request.format == "json":
        payload["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": "lead-classification",
                "schema": OUTPUT_SCHEMA,
            },
        }

    try:
        async with httpx.AsyncClient(
            timeout=VLLM_TIMEOUT
        ) as client:
            response = await client.post(
                f"{VLLM_HOST}/v1/chat/completions",
                json=payload,
            )
            response.raise_for_status()

    except httpx.TimeoutException as exc:
        raise HTTPException(
            status_code=504,
            detail="vLLM timeout",
        ) from exc

    except httpx.HTTPStatusError as exc:
        error_body = exc.response.text

        logger.error(
            "vLLM HTTP error %s: %s",
            exc.response.status_code,
            error_body,
        )

        raise HTTPException(
            status_code=503,
            detail=(
                "vLLM HTTP error: "
                f"{exc.response.status_code}; "
                f"{error_body[:500]}"
            ),
        ) from exc

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to vLLM: {exc}",
        ) from exc

    result = response.json()

    choices = result.get("choices") or []

    if not choices:
        raise HTTPException(
            status_code=502,
            detail="vLLM returned no choices",
        )

    message = choices[0].get("message") or {}
    raw = message.get("content") or ""

    print(
        {
            "model": result.get("model"),
            "response": raw,
            "reasoning": message.get("reasoning_content"),
            "finish_reason": choices[0].get(
                "finish_reason"
            ),
            "usage": result.get("usage"),
        },
        flush=True,
    )

    if request.format == "json":
        try:
            data = extract_json(raw)

            return GenerateResponse(
                ok=True,
                data=data,
                raw=raw,
            )

        except Exception as exc:
            logger.warning(
                "Invalid JSON from model: %s",
                raw,
            )

            return GenerateResponse(
                ok=False,
                data=None,
                raw=raw,
                error=f"Invalid JSON from model: {exc}",
            )

    return GenerateResponse(
        ok=True,
        data=None,
        raw=raw,
    )