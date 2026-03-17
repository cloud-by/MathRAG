from __future__ import annotations

from typing import Any, Dict, List

from fastapi import APIRouter, HTTPException, status
from openai import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from app.schemas.chat import ChatRequest, ChatResponse
from app.services.rag_pipeline import chat_with_rag


router = APIRouter(prefix="/api", tags=["chat"])


def _history_to_dicts(history: List[Any]) -> List[Dict[str, str]]:
    normalized_history: List[Dict[str, str]] = []

    for item in history:
        role = str(getattr(item, "role", "")).strip().lower()
        content = str(getattr(item, "content", "")).strip()

        if not role or not content:
            continue

        normalized_history.append(
            {
                "role": role,
                "content": content,
            }
        )

    return normalized_history


@router.post("/chat", response_model=ChatResponse, summary="数学 RAG 问答")
def chat(request: ChatRequest) -> ChatResponse:
    try:
        result = chat_with_rag(
            question=request.question,
            history=_history_to_dicts(request.history),
            top_k=request.top_k,
        )

        if not isinstance(result, dict):
            raise ValueError("RAG 管道返回结果格式错误，应为对象")

        return ChatResponse(**result)

    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc

    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"系统文件缺失：{exc}",
        ) from exc

    except AuthenticationError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="大模型 API 鉴权失败，请检查 LLM_API_KEY。",
        ) from exc

    except RateLimitError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="大模型 API 触发限流，请稍后重试。",
        ) from exc

    except APITimeoutError as exc:
        raise HTTPException(
            status_code=status.HTTP_504_GATEWAY_TIMEOUT,
            detail="大模型 API 请求超时。",
        ) from exc

    except APIConnectionError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail="无法连接到大模型 API，请检查网络、API 服务状态或 base_url。",
        ) from exc

    except APIStatusError as exc:
        message = "大模型 API 返回错误。"

        try:
            error_obj = getattr(exc, "response", None)
            if error_obj is not None:
                payload = error_obj.json()
                if isinstance(payload, dict):
                    error_info = payload.get("error", {})
                    if isinstance(error_info, dict):
                        message = error_info.get("message") or message
        except Exception:
            pass

        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=message,
        ) from exc

    except APIError as exc:
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"大模型 API 调用失败：{exc}",
        ) from exc

    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"系统内部错误：{exc}",
        ) from exc