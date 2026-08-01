from dataclasses import dataclass
from typing import Literal
from uuid import UUID


UserRole = Literal["student", "teacher", "admin"]
UserStatus = Literal["active", "disabled"]
USER_ROLES = frozenset({"student", "teacher", "admin"})
USER_STATUSES = frozenset({"active", "disabled"})


@dataclass(frozen=True)
class UserActor:
    user_id: UUID
    role: UserRole
