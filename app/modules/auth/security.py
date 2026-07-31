"""密码、Session 令牌和 CSRF 的安全原语。"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import secrets

from pwdlib import PasswordHash


SESSION_TOKEN_BYTES = 32
_password_hash = PasswordHash.recommended()


async def hash_password(password: str) -> str:
    """在线程池中执行 Argon2 密码哈希。"""
    return await asyncio.to_thread(_password_hash.hash, password)


def _safe_verify(password: str, encoded_hash: str) -> bool:
    try:
        return _password_hash.verify(password, encoded_hash)
    except Exception:
        return False


async def verify_password(password: str, encoded_hash: str) -> bool:
    """验证密码；未知或损坏的哈希统一视为不匹配。"""
    return await asyncio.to_thread(_safe_verify, password, encoded_hash)


def generate_session_token() -> str:
    """生成只允许短暂存在于应用内存和 Cookie 中的随机令牌。"""
    return secrets.token_urlsafe(SESSION_TOKEN_BYTES)


def hash_session_token(token: str) -> bytes:
    """生成数据库可持久化的固定长度 Session 摘要。"""
    return hashlib.sha256(token.encode("utf-8")).digest()


def issue_csrf_token(session_hash: bytes, secret: str) -> str:
    """签发与服务端 Session 摘要绑定的 CSRF token。"""
    nonce = secrets.token_urlsafe(32)
    signature = hmac.new(
        secret.encode("utf-8"),
        session_hash + b"." + nonce.encode("ascii"),
        hashlib.sha256,
    ).hexdigest()
    return f"{nonce}.{signature}"


def verify_csrf_token(token: str, session_hash: bytes, secret: str) -> bool:
    """校验已有 CSRF token，不生成新 nonce，也不抛出解析错误。"""
    try:
        nonce, supplied_signature = token.split(".", 1)
        nonce_bytes = nonce.encode("ascii")
        if not nonce or not supplied_signature:
            return False
        expected_signature = hmac.new(
            secret.encode("utf-8"),
            session_hash + b"." + nonce_bytes,
            hashlib.sha256,
        ).hexdigest()
    except (UnicodeError, ValueError):
        return False
    return hmac.compare_digest(supplied_signature, expected_signature)
