from __future__ import annotations

from fastapi import APIRouter, HTTPException, status
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.services.knowledge_extractor import DEFAULT_KNOWLEDGE_PATH, append_records, extract_knowledge_records
from app.schemas.knowledge import KnowledgeExtractRequest, KnowledgeExtractResponse


router = APIRouter(prefix="/api/knowledge", tags=["knowledge"])


@router.post("/extract", response_model=KnowledgeExtractResponse, summary="Extract knowledge records from textbook text")
def extract_knowledge(request: KnowledgeExtractRequest) -> KnowledgeExtractResponse:
    try:
        records = extract_knowledge_records(
            text=request.text,
            category=request.category,
        )
        saved_count = append_records(records) if request.save else 0

        next_steps = []
        if request.save:
            next_steps = [
                "Run: python -m scripts.build_kb",
                "Run: python -m scripts.build_index",
                "Restart the service if it is already running.",
            ]

        return KnowledgeExtractResponse(
            records=records,
            saved_count=saved_count,
            knowledge_path=str(DEFAULT_KNOWLEDGE_PATH),
            next_steps=next_steps,
        )

    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="LLM API authentication failed. Check LLM_API_KEY.",
        ) from exc

    except RateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="LLM API rate limit was reached. Try again later.",
        ) from exc

    except APITimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="LLM API request timed out.",
        ) from exc

    except APIConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="Could not connect to the LLM API. Check the network and LLM_BASE_URL.",
        ) from exc

    except APIStatusError as exc:
        message = "LLM API returned an error."
        try:
            payload = exc.response.json()
            if isinstance(payload, dict):
                error_info = payload.get("error", {})
                if isinstance(error_info, dict):
                    message = error_info.get("message") or message
        except Exception:
            pass
        raise HTTPException(status_code=status.HTTP_502_BAD_GATEWAY, detail=message) from exc

    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"LLM API call failed: {exc}",
        ) from exc
