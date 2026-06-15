from __future__ import annotations

from typing import Any, List, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator


VALID_DIFFICULTIES = {"easy", "medium", "hard"}
SEED_FIELD_ORDER = [
    "id",
    "category",
    "title",
    "keywords",
    "content",
    "example",
    "steps",
    "difficulty",
]


class KnowledgeRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str = Field(..., description="Knowledge item id, for example k0001")
    category: str = Field(..., description="Knowledge category")
    title: str = Field(..., description="Knowledge point title")
    keywords: List[str] = Field(default_factory=list, description="Keywords")
    content: str = Field(..., description="Core explanation")
    example: str = Field(default="", description="Example or application")
    steps: List[str] = Field(default_factory=list, description="Understanding or solving steps")
    difficulty: str = Field(..., description="easy / medium / hard")

    @field_validator("id", "category", "title", "content", "example", "difficulty", mode="before")
    @classmethod
    def normalize_text(cls, value: Any) -> str:
        return "" if value is None else str(value).strip()

    @field_validator("keywords", "steps", mode="before")
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

    @field_validator("id")
    @classmethod
    def validate_id(cls, value: str) -> str:
        if not value.startswith("k") or not value[1:].isdigit():
            raise ValueError("id must use k0001 style")
        return value

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, value: str) -> str:
        if value not in VALID_DIFFICULTIES:
            raise ValueError("difficulty must be easy, medium, or hard")
        return value

    @field_validator("category", "title", "content")
    @classmethod
    def validate_required_text(cls, value: str) -> str:
        if not value:
            raise ValueError("required text field cannot be empty")
        return value

    @field_validator("keywords", "steps")
    @classmethod
    def validate_required_lists(cls, value: List[str]) -> List[str]:
        if not value:
            raise ValueError("keywords and steps cannot be empty")
        return value

    def to_seed_dict(self) -> dict[str, Any]:
        data = self.model_dump()
        return {field: data[field] for field in SEED_FIELD_ORDER}


class KnowledgeExtractRequest(BaseModel):
    text: str = Field(..., description="Text excerpt from a math textbook")
    category: Optional[str] = Field(default=None, description="Optional category hint")
    save: bool = Field(default=True, description="Whether to append records to the raw knowledge JSONL file")

    @field_validator("text")
    @classmethod
    def validate_text(cls, value: str) -> str:
        value = str(value or "").strip()
        if not value:
            raise ValueError("text cannot be empty")
        return value

    @field_validator("category")
    @classmethod
    def normalize_optional_text(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = str(value).strip()
        return value or None


class KnowledgeExtractResponse(BaseModel):
    records: List[KnowledgeRecord]
    saved_count: int = 0
    knowledge_path: str
    next_steps: List[str] = Field(default_factory=list)
