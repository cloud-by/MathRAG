from __future__ import annotations

import uvicorn

from app.core.config import settings


def main() -> None:
    """Application entry point."""
    uvicorn.run(
        "app.main:app",
        host=settings.APP_HOST,
        port=settings.APP_PORT,
        reload=getattr(settings, "APP_DEBUG", True),
    )


if __name__ == "__main__":
    main()
