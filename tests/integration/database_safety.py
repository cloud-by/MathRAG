"""破坏性集成测试的数据库身份守卫。"""

from __future__ import annotations

from sqlalchemy.engine import URL, make_url


EXPECTED_TEST_DATABASE = "mathrag_test"


def require_test_database_url(test_database_url: str, database_url: str | None = None) -> str:
    """确认破坏性测试仅会访问专用测试库，返回原始测试 URL。"""
    test_url = _parse_url(test_database_url, "TEST_DATABASE_URL")
    if not test_url.database:
        raise RuntimeError("TEST_DATABASE_URL 必须包含数据库名")
    if test_url.database != EXPECTED_TEST_DATABASE:
        raise RuntimeError(
            "TEST_DATABASE_URL 必须指向数据库 "
            f"{EXPECTED_TEST_DATABASE}，实际数据库名为 {test_url.database}"
        )

    if database_url:
        application_url = _parse_url(database_url, "DATABASE_URL")
        if _connection_identity(test_url) == _connection_identity(application_url):
            raise RuntimeError("TEST_DATABASE_URL 不得与 DATABASE_URL 指向同一数据库")

    return test_database_url


def _parse_url(database_url: str, variable_name: str) -> URL:
    """解析连接 URL，解析异常不携带原始地址或密码。"""
    try:
        return make_url(database_url)
    except Exception as error:
        raise RuntimeError(f"{variable_name} 格式无效") from error


def _connection_identity(url: URL) -> tuple[
    str,
    str | None,
    int | None,
    str | None,
    str | None,
    tuple[tuple[str, tuple[str, ...]], ...],
]:
    """提取不含密码的规范化连接身份，用于拒绝同库误配。"""
    try:
        port = url.port
    except ValueError as error:
        raise RuntimeError("数据库连接端口无效") from error

    default_port = 5432 if url.get_backend_name() == "postgresql" else None
    return (
        url.get_backend_name(),
        url.host.lower() if url.host else None,
        port if port is not None else default_port,
        url.username,
        url.database,
        tuple(sorted(url.normalized_query.items())),
    )
