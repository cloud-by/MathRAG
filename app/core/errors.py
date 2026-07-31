class ConfigurationError(RuntimeError):
    """表示应用运行配置不完整或无效。"""


class AppError(Exception):
    """表示可安全返回给 API 调用方的应用错误。"""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int,
        details: dict[str, object] | None = None,
    ) -> None:
        super().__init__(code)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.details = {} if details is None else details
