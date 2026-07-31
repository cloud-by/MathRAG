"""认证安全原语测试。"""

from __future__ import annotations

import asyncio

from app.modules.auth.security import (
    generate_session_token,
    hash_password,
    hash_session_token,
    issue_csrf_token,
    verify_csrf_token,
    verify_password,
)


def test_argon2_hash_uses_independent_salts_and_never_contains_password() -> None:
    password = "correct horse battery staple"

    first = asyncio.run(hash_password(password))
    second = asyncio.run(hash_password(password))

    assert first != second
    assert password not in first
    assert first.startswith("$argon2")
    assert asyncio.run(verify_password(password, first)) is True
    assert asyncio.run(verify_password("wrong-password", first)) is False


def test_verify_password_treats_unknown_hash_as_non_match() -> None:
    assert asyncio.run(verify_password("private-password", "not-a-password-hash")) is False


def test_session_token_hash_is_stable_sha256_without_raw_token() -> None:
    token = generate_session_token()
    digest = hash_session_token(token)

    assert len(digest) == 32
    assert digest == hash_session_token(token)
    assert token.encode("utf-8") not in digest


def test_csrf_token_is_bound_to_session_hash_and_secret() -> None:
    session_hash = hash_session_token(generate_session_token())
    other_session_hash = hash_session_token(generate_session_token())
    token = issue_csrf_token(session_hash, "s" * 32)

    assert verify_csrf_token(token, session_hash, "s" * 32) is True
    assert verify_csrf_token(token, other_session_hash, "s" * 32) is False
    assert verify_csrf_token(token, session_hash, "x" * 32) is False
    assert verify_csrf_token("invalid", session_hash, "s" * 32) is False
    assert verify_csrf_token("bad-nonce.not-hex", session_hash, "s" * 32) is False
