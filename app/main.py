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

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://ollama:11434")
DEFAULT_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:7b")
OLLAMA_TIMEOUT = float(os.getenv("OLLAMA_TIMEOUT", "60"))


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


def extract_json(raw: str) -> dict[str, Any]:
    raw = raw.strip()

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)

    if not match:
        raise ValueError(f"No JSON found in model response: {raw}")

    return json.loads(match.group(0))


@app.get("/health")
async def health():
    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f"{OLLAMA_HOST}/api/tags")
            response.raise_for_status()

        return {
            "status": "ok",
            "ollama": "available",
            "default_model": DEFAULT_MODEL,
        }

    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama unavailable: {exc}",
        )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    model = request.model or DEFAULT_MODEL

    payload = {
        "model": model,
        "prompt": request.prompt,
        "stream": False,
        "options": {
            "temperature": request.temperature,
        },
    }

    if request.format:
        payload["format"] = request.format

    try:
        async with httpx.AsyncClient(timeout=OLLAMA_TIMEOUT) as client:
            response = await client.post(
                f"{OLLAMA_HOST}/api/generate",
                json=payload,
            )
            response.raise_for_status()

    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Ollama timeout")

    except httpx.HTTPStatusError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Ollama HTTP error: {exc.response.status_code}",
        )

    except httpx.RequestError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Cannot connect to Ollama: {exc}",
        )

    result = response.json()
    raw = result.get("response", "")

    if request.format == "json":
        try:
            data = extract_json(raw)

            return GenerateResponse(
                ok=True,
                data=data,
                raw=raw,
            )

        except Exception as exc:
            logger.warning("Invalid JSON from model: %s", raw)

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