"""会话 Repository SQL 与事务边界测试。"""

from __future__ import annotations

import ast
from pathlib import Path


REPOSITORY_PATH = (
    Path(__file__).resolve().parents[3]
    / "app"
    / "modules"
    / "conversations"
    / "repository.py"
)


def test_repository_source_contains_owner_scope_for_resource_and_message_queries() -> None:
    source = REPOSITORY_PATH.read_text(encoding="utf-8")

    assert source.count("Conversation.user_id == user_id") >= 5
    assert "Message.conversation_id == Conversation.id" in source
    assert "Conversation.id == conversation_id" in source


def test_repository_does_not_control_injected_session_lifecycle() -> None:
    tree = ast.parse(REPOSITORY_PATH.read_text(encoding="utf-8"))
    forbidden = {
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Attribute)
        and isinstance(node.func.value.value, ast.Name)
        and node.func.value.value.id == "self"
        and node.func.value.attr == "_session"
        and node.func.attr in {"begin", "commit", "rollback", "close"}
    }

    assert forbidden == set()
