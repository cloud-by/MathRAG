"""会话 API schema 测试。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.modules.conversations.schemas import ConversationCreate, ConversationUpdate


def test_conversation_title_collapses_whitespace() -> None:
    assert ConversationCreate(title="  导数   复习\n计划 ").title == "导数 复习 计划"
    assert ConversationUpdate(title="  新   标题 ").title == "新 标题"


@pytest.mark.parametrize("title", ["", "   ", "x" * 256])
def test_conversation_title_rejects_empty_or_too_long_value(title: str) -> None:
    with pytest.raises(ValidationError):
        ConversationCreate(title=title)


def test_conversation_update_rejects_unknown_fields_and_invalid_status() -> None:
    with pytest.raises(ValidationError):
        ConversationUpdate(status="deleted")
    with pytest.raises(ValidationError):
        ConversationUpdate(user_id="not-allowed")
