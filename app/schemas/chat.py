from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


class ChatTurn(BaseModel):
    role: str = Field(..., description="消息角色，通常为 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = str(value).strip().lower()
        if value not in {"user", "assistant", "system"}:
            raise ValueError("role 必须是 user、assistant 或 system")
        return value

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("content 不能为空")
        return value


class ChatRequest(BaseModel):
    question: str = Field(..., description="用户当前问题")
    history: List[ChatTurn] = Field(default_factory=list, description="最近几轮对话历史")
    top_k: Optional[int] = Field(default=None, ge=1, le=10, description="检索返回的参考知识条数")

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        value = str(value).strip()
        if not value:
            raise ValueError("question 不能为空")
        return value


class ReferenceItem(BaseModel):
    rank: int
    score: float
    index: Optional[int] = None
    chunk_id: str
    source_id: str
    title: str
    category: str
    keywords: List[str] = Field(default_factory=list)
    content: str = ""
    example: str = ""
    steps: List[str] = Field(default_factory=list)
    answer_context: str = ""
    metadata: Dict[str, Any] = Field(default_factory=dict)


class ChatResponse(BaseModel):
    question: str
    answer: str
    steps: List[str] = Field(default_factory=list)
    used_knowledge: List[str] = Field(default_factory=list)
    related_questions: List[str] = Field(default_factory=list)
    references: List[ReferenceItem] = Field(default_factory=list)
    reasoning_content: Optional[str] = None


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str
