from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator


VALID_ROLES = {"user", "assistant", "system"}
VALID_STAGES = {"primary", "junior_secondary", "senior_secondary", "undergraduate"}
VALID_DIFFICULTIES = {"easy", "medium", "hard"}


class ChatTurn(BaseModel):
    role: str = Field(..., description="消息角色，通常为 user 或 assistant")
    content: str = Field(..., description="消息内容")

    @field_validator("role")
    @classmethod
    def validate_role(cls, value: str) -> str:
        value = str(value).strip().lower()
        if value not in VALID_ROLES:
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
    rank: int = Field(..., description="检索排序名次，从 1 开始")
    score: float = Field(..., description="向量检索得分")
    index: Optional[int] = Field(default=None, description="FAISS 内部索引位置")

    chunk_id: str = Field(..., description="chunk 唯一标识")
    source_id: str = Field(..., description="原始知识点 id")

    category: str = Field(..., description="知识类别")
    stage: str = Field(..., description="学段：primary / junior_secondary / senior_secondary / undergraduate")
    course: str = Field(..., description="课程名称")
    title: str = Field(..., description="知识点标题")

    keywords: List[str] = Field(default_factory=list, description="关键词列表")
    content: str = Field(default="", description="知识点核心内容")
    example: str = Field(default="", description="示例或应用场景")
    steps: List[str] = Field(default_factory=list, description="理解或解题步骤")
    prerequisites: List[str] = Field(default_factory=list, description="前置知识列表")
    difficulty: str = Field(..., description="难度：easy / medium / hard")

    answer_context: str = Field(default="", description="面向回答构造的上下文文本")
    retrieval_text: str = Field(default="", description="面向向量检索的拼接文本")
    source_line: Optional[int] = Field(default=None, description="原始 JSONL 所在行号")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="附加元数据")

    @field_validator(
        "chunk_id",
        "source_id",
        "category",
        "stage",
        "course",
        "title",
        "difficulty",
        "content",
        "example",
        "answer_context",
        "retrieval_text",
        mode="before",
    )
    @classmethod
    def normalize_text_fields(cls, value: Any) -> str:
        if value is None:
            return ""
        return str(value).strip()

    @field_validator("stage")
    @classmethod
    def validate_stage(cls, value: str) -> str:
        if value not in VALID_STAGES:
            raise ValueError(
                "stage 必须是 primary、junior_secondary、senior_secondary 或 undergraduate"
            )
        return value

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: str) -> str:
        if value not in VALID_DIFFICULTIES:
            raise ValueError("difficulty 必须是 easy、medium 或 hard")
        return value

    @field_validator("keywords", "steps", "prerequisites", mode="before")
    @classmethod
    def normalize_str_list(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            output: List[str] = []
            seen = set()
            for item in value:
                text = str(item).strip()
                if text and text not in seen:
                    output.append(text)
                    seen.add(text)
            return output

        text = str(value).strip()
        return [text] if text else []

    @field_validator("rank")
    @classmethod
    def validate_rank(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("rank 必须大于 0")
        return int(value)

    @field_validator("score")
    @classmethod
    def validate_score(cls, value: float) -> float:
        return float(value)

    @field_validator("index")
    @classmethod
    def validate_index(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        return int(value)

    @field_validator("source_line")
    @classmethod
    def validate_source_line(cls, value: Optional[int]) -> Optional[int]:
        if value is None:
            return None
        value = int(value)
        if value <= 0:
            raise ValueError("source_line 必须大于 0")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def validate_metadata(cls, value: Any) -> Dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("metadata 必须是对象")
        return value


class ChatResponse(BaseModel):
    question: str = Field(..., description="用户当前问题")
    answer: str = Field(..., description="最终回答")
    steps: List[str] = Field(default_factory=list, description="回答步骤")
    used_knowledge: List[str] = Field(default_factory=list, description="本次实际使用到的知识点标题")
    related_questions: List[str] = Field(default_factory=list, description="推荐追问")
    references: List[ReferenceItem] = Field(default_factory=list, description="检索到的参考知识")
    reasoning_content: Optional[str] = Field(default=None, description="模型额外的推理内容（可选）")

    @field_validator("question", "answer", mode="before")
    @classmethod
    def normalize_required_text(cls, value: Any) -> str:
        value = "" if value is None else str(value).strip()
        if not value:
            raise ValueError("question 和 answer 不能为空")
        return value

    @field_validator("steps", "used_knowledge", "related_questions", mode="before")
    @classmethod
    def normalize_response_lists(cls, value: Any) -> List[str]:
        if value is None:
            return []
        if isinstance(value, list):
            output: List[str] = []
            seen = set()
            for item in value:
                text = str(item).strip()
                if text and text not in seen:
                    output.append(text)
                    seen.add(text)
            return output

        text = str(value).strip()
        return [text] if text else []


class HealthResponse(BaseModel):
    status: str = "ok"
    app_name: str