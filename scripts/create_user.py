"""通过安全交互输入创建 MathRAG 用户。"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import sys
from collections.abc import Sequence

from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

from app.core.errors import AppError
from app.infrastructure.database.session import get_session_factory
from app.modules.users.repository import UserRepository
from app.modules.users.service import UserService


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="创建 MathRAG 用户")
    parser.add_argument("--username", required=True)
    parser.add_argument("--email")
    parser.add_argument("--role", choices=("admin", "user"), default="user")
    return parser


async def _create_user(
    args: argparse.Namespace,
    password: str,
    session_factory: async_sessionmaker[AsyncSession],
) -> dict[str, str]:
    async with session_factory() as session:
        async with session.begin():
            service = UserService(UserRepository(session))
            user = await service.create_user(
                username=args.username,
                password=password,
                email=args.email,
                role=args.role,
            )
            await session.flush()
    return {
        "id": str(user.id),
        "username": user.username,
        "role": user.role,
        "status": "USER_CREATED",
    }


def main(
    argv: Sequence[str] | None = None,
    *,
    session_factory: async_sessionmaker[AsyncSession] | None = None,
) -> int:
    args = build_parser().parse_args(argv)
    password = getpass.getpass("密码: ")
    confirmation = getpass.getpass("确认密码: ")
    if password != confirmation:
        print("USER_INPUT_INVALID: 两次密码输入不一致。", file=sys.stderr)
        return 2

    try:
        result = asyncio.run(
            _create_user(
                args,
                password,
                session_factory or get_session_factory(),
            )
        )
    except AppError as exc:
        print(f"{exc.code}: {exc.message}", file=sys.stderr)
        return 2
    except Exception:
        print("USER_CREATE_FAILED", file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
