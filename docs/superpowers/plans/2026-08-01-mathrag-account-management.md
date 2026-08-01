# MathRAG 完整账号管理 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在现有 MathRAG 认证体系中加入学生、教师、管理员三种角色，使管理员能够管理全部账号，教师只能完整管理自己创建的学生，并强制新账号修改临时密码。

**Architecture:** 保留单一 `users` 表，通过 `created_by_user_id` 自关联表达不可转移的创建归属，通过 `must_change_password` 强制临时密码闭环。用户管理权限集中在 `UserService`，仓储查询同时施加数据范围；FastAPI 依赖负责认证、CSRF 和角色入口，Vue 使用生成的 OpenAPI 类型和路由守卫提供管理界面。

**Tech Stack:** Python 3.11、FastAPI、Pydantic 2、SQLAlchemy 2 Async、PostgreSQL 18、Alembic、Argon2、pytest、Vue 3.5、TypeScript 5.9、Vue Router、Vitest、Testing Library、Playwright、Docker Compose。

---

## 文件结构

新增后端文件：

- `alembic/versions/0006_add_account_management.py`：角色转换、创建者自关联和临时密码字段迁移。
- `app/modules/users/types.py`：角色、状态和操作者类型的唯一来源。
- `app/modules/users/dependencies.py`：在一个事务中装配用户服务、仓储和 Session 撤销器。
- `app/modules/users/router.py`：用户列表、创建、详情、编辑和密码重置接口。
- `tests/integration/test_account_management_migration.py`：0006 数据与模式契约。
- `tests/api/test_users.py`：用户管理 API、权限和 CSRF 契约。

新增前端文件：

- `frontend/src/features/auth/ChangePasswordPage.vue`：临时密码强制修改页。
- `frontend/src/features/users/types.ts`、`api.ts`、`useUsers.ts`：用户管理类型、client 和查询状态。
- `frontend/src/features/users/UserListPage.vue`：筛选、分页和账号列表。
- `frontend/src/features/users/UserEditorPage.vue`：创建与编辑共用页。
- `frontend/src/features/users/ResetPasswordDialog.vue`：密码重置确认表单。
- `frontend/src/features/users/users.spec.ts`：用户管理单元测试。
- `frontend/tests/e2e/account-management.spec.ts`：三角色与临时密码 E2E。

重点修改现有文件：

- `app/modules/users/models.py`、`schemas.py`、`repository.py`、`service.py`：持久化、DTO、范围查询和权限。
- `app/modules/auth/repository.py`、`service.py`、`schemas.py`、`dependencies.py`、`router.py`：新角色、临时密码 principal 和修改密码。
- `app/modules/conversations/router.py`、`app/modules/knowledge/router.py`、`app/modules/rag/router.py`：业务端点改用“密码已就绪”依赖。
- `app/main.py`：注册用户路由。
- `scripts/create_user.py`：三角色 CLI。
- `frontend/src/features/auth/api.ts`、`frontend/src/features/auth/useAuth.ts`、`frontend/src/features/auth/LoginPage.vue`：修改密码闭环。
- `frontend/src/router/{index,meta.d}.ts`、`frontend/src/app/AppNavigation.vue`：守卫和角色导航。
- `frontend/openapi.json`、`frontend/src/api/schema.d.ts`：生成契约。
- `frontend/tests/e2e/fixtures.ts`、`README.md`：mock 与使用说明。

## Task 1：迁移角色、创建归属与临时密码字段

**Files:**
- Create: `alembic/versions/0006_add_account_management.py`
- Create: `tests/integration/test_account_management_migration.py`
- Create: `app/modules/users/types.py`
- Modify: `app/modules/users/models.py`
- Modify: `tests/integration/test_migrations.py`
- Modify: `tests/integration/test_m4_migration_schema.py`
- Modify: `tests/integration/test_m5_migration_schema.py`

- [ ] **Step 1：先写迁移失败测试**

在 0005 边界插入旧 `user/admin`，升级后验证数据转换和模式：

```python
def test_account_management_migration_converts_existing_roles() -> None:
    database_url = require_test_database_url(
        os.environ["TEST_DATABASE_URL"], os.getenv("DATABASE_URL")
    )
    try:
        run_alembic(database_url, "downgrade", "0005_create_documents_ingestion_jobs")
        student_id, admin_id = asyncio.run(seed_legacy_users(database_url))
        run_alembic(database_url, "upgrade", "0006_add_account_management")
        rows, columns, constraints, indexes = asyncio.run(fetch_contract(database_url))
        assert rows[student_id]["role"] == "student"
        assert rows[admin_id]["role"] == "admin"
        assert rows[student_id]["created_by_user_id"] is None
        assert rows[student_id]["must_change_password"] is False
        assert constraints["ck_users_role"] == {"student", "teacher", "admin"}
        assert constraints["fk_users_created_by_user_id_users"] == "ON DELETE SET NULL"
        assert "ix_users_created_by_user_id" in indexes
        assert "student" in columns["role"]["column_default"]
    finally:
        run_alembic(database_url, "upgrade", "head")
```

同文件增加：删除创建者后外键置空；存在 `teacher` 时 downgrade 明确失败，删除教师后 `student` 可恢复为 `user`。

- [ ] **Step 2：运行测试并确认缺少 revision**

```powershell
docker compose run --rm app pytest -q tests/integration/test_account_management_migration.py
```

Expected: FAIL，`0006_add_account_management` 不存在。

- [ ] **Step 3：增加共享类型和 ORM 字段**

创建 `app/modules/users/types.py`：

```python
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
```

在 `models.py` 加入自关联、布尔标记和新默认值：

```python
ForeignKeyConstraint(
    ["created_by_user_id"], ["users.id"],
    name="fk_users_created_by_user_id_users", ondelete="SET NULL",
),
CheckConstraint("role IN ('student', 'teacher', 'admin')", name="role"),
Index("ix_users_created_by_user_id", "created_by_user_id"),

created_by_user_id: Mapped[UUID | None] = mapped_column(
    PostgreSQLUUID(as_uuid=True), nullable=True
)
must_change_password: Mapped[bool] = mapped_column(
    Boolean(), nullable=False, default=False, server_default=false()
)
role: Mapped[str] = mapped_column(
    String(16), nullable=False, default="student", server_default=text("'student'")
)
```

- [ ] **Step 4：实现自包含 migration**

`0006_add_account_management.py` 只导入 Alembic、SQLAlchemy 和 PostgreSQL dialect：

```python
revision: str = "0006_add_account_management"
down_revision: str | None = "0005_create_documents_ingestion_jobs"


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column("created_by_user_id", postgresql.UUID(as_uuid=True), nullable=True),
    )
    op.add_column(
        "users",
        sa.Column("must_change_password", sa.Boolean(), server_default=sa.false(), nullable=False),
    )
    op.execute(sa.text("UPDATE users SET role = 'student' WHERE role = 'user'"))
    op.drop_constraint("ck_users_role", "users", type_="check")
    op.create_check_constraint(
        "ck_users_role", "users", "role IN ('student', 'teacher', 'admin')"
    )
    op.alter_column("users", "role", server_default="student")
    op.create_foreign_key(
        "fk_users_created_by_user_id_users", "users", "users",
        ["created_by_user_id"], ["id"], ondelete="SET NULL",
    )
    op.create_index("ix_users_created_by_user_id", "users", ["created_by_user_id"])


def downgrade() -> None:
    teacher_count = op.get_bind().scalar(
        sa.text("SELECT count(*) FROM users WHERE role = 'teacher'")
    )
    if teacher_count:
        raise RuntimeError("存在 teacher 账号，无法降级账号管理迁移。")
    op.drop_index("ix_users_created_by_user_id", table_name="users")
    op.drop_constraint("fk_users_created_by_user_id_users", "users", type_="foreignkey")
    op.execute(sa.text("UPDATE users SET role = 'user' WHERE role = 'student'"))
    op.drop_constraint("ck_users_role", "users", type_="check")
    op.create_check_constraint("ck_users_role", "users", "role IN ('admin', 'user')")
    op.alter_column("users", "role", server_default="user")
    op.drop_column("users", "must_change_password")
    op.drop_column("users", "created_by_user_id")
```

- [ ] **Step 5：更新 head 断言并运行迁移组**

四个迁移测试的最终断言统一为：

```python
assert "0006_add_account_management (head)" in current.stdout
```

```powershell
docker compose run --rm app pytest -q tests/integration/test_account_management_migration.py tests/integration/test_migrations.py tests/integration/test_m4_migration_schema.py tests/integration/test_m5_migration_schema.py
```

Expected: PASS，测试结束后数据库回到 0006 head。

- [ ] **Step 6：提交**

```powershell
git add alembic/versions/0006_add_account_management.py app/modules/users/types.py app/modules/users/models.py tests/integration/test_account_management_migration.py tests/integration/test_migrations.py tests/integration/test_m4_migration_schema.py tests/integration/test_m5_migration_schema.py
git commit -m "feat: migrate account roles and ownership"
```

## Task 2：实现范围查询和用户权限服务

**Files:**
- Modify: `app/modules/users/schemas.py`
- Modify: `app/modules/users/repository.py`
- Modify: `app/modules/users/service.py`
- Modify: `tests/unit/auth/test_user_service.py`
- Modify: `tests/integration/test_user_repository.py`

- [ ] **Step 1：先写权限矩阵失败测试**

在 `test_user_service.py` 加入教师隔离和管理员保护：

```python
def test_teacher_manages_only_students_created_by_self() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    teacher = add_user(repository, role="teacher")
    own = add_user(repository, role="student", created_by_user_id=teacher.id)
    other = add_user(repository, role="student", created_by_user_id=uuid4())
    actor = UserActor(teacher.id, "teacher")

    updated = asyncio.run(
        service.update_managed_user(
            actor, own.id, UserUpdate(email="new@example.local"), NOW
        )
    )
    assert updated.email == "new@example.local"
    with pytest.raises(AppError) as exc_info:
        asyncio.run(service.get_managed_user(actor, other.id))
    assert exc_info.value.code == "USER_NOT_FOUND"
    assert exc_info.value.status_code == 404


def test_admin_cannot_disable_self() -> None:
    repository = FakeUserRepository()
    service = UserService(repository, FakeSessionRevoker())
    admin = add_user(repository, role="admin", status="active")
    with pytest.raises(AppError) as exc_info:
        asyncio.run(
            service.update_managed_user(
                UserActor(admin.id, "admin"),
                admin.id,
                UserUpdate(status="disabled"),
                NOW,
            )
        )
    assert exc_info.value.code == "USER_SELF_PROTECTED"
```

同组覆盖：管理员创建三种角色；教师只能创建学生且不能传 role 更新；教师列表仅返回自己的学生；禁用、角色变化和重置密码撤销 Session；最后一个启用管理员受保护。

- [ ] **Step 2：确认测试失败**

```powershell
docker compose run --rm app pytest -q tests/unit/auth/test_user_service.py
```

Expected: FAIL，缺少管理 DTO 和 actor 驱动方法。

- [ ] **Step 3：定义请求与响应 DTO**

在 `schemas.py` 增加：

```python
class ManagedUserRead(UserRead):
    created_by_username: str | None


class UserPage(BaseModel):
    model_config = ConfigDict(frozen=True)
    items: list[ManagedUserRead]
    page: int
    page_size: int
    total: int


class UserCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str = Field(min_length=1, max_length=64)
    email: str | None = Field(default=None, max_length=320)
    password: str = Field(min_length=12, max_length=128)
    role: UserRole = "student"


class UserUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    username: str | None = Field(default=None, min_length=1, max_length=64)
    email: str | None = Field(default=None, max_length=320)
    role: UserRole | None = None
    status: UserStatus | None = None

    @model_validator(mode="after")
    def require_change(self) -> "UserUpdate":
        if not self.model_fields_set:
            raise ValueError("至少提供一个需要修改的字段")
        return self


class UserPasswordReset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    password: str = Field(min_length=12, max_length=128)
```

`UserRead` 同步增加 `created_by_user_id`、`must_change_password`，并把 role/status 标注为共享类型。现有 CLI 使用的 `create_user` 默认值同步固定为 `role="student"`、`created_by_user_id=None`、`must_change_password=True`；只有迁移前已存在的账号保留 false。

- [ ] **Step 4：实现 SQL 范围和锁**

`repository.py` 的教师条件必须写入 SQL：

```python
def _visible_clause(actor: UserActor):
    if actor.role == "admin":
        return true()
    return and_(
        User.role == "student",
        User.created_by_user_id == actor.user_id,
    )


async def lock_active_admins(self) -> list[User]:
    result = await self._session.scalars(
        select(User)
        .where(User.role == "admin", User.status == "active")
        .order_by(User.id)
        .with_for_update()
    )
    return list(result)
```

`list_managed` 的 items/count 使用同一条件，支持 `q/role/status/page/page_size`，排序固定为 `created_at DESC, id DESC`。`get_managed_by_id(actor, user_id, for_update=True)` 使用 `aliased(User)` outer join 创建者用户名并锁定目标 User。

- [ ] **Step 5：实现服务策略**

服务核心规则：

```python
async def create_managed_user(self, actor: UserActor, request: UserCreate) -> ManagedUserRead:
    if actor.role == "teacher" and request.role != "student":
        raise _error("USER_ROLE_FORBIDDEN", "教师只能创建学生账号。", 403)
    return await self._create(
        username=request.username,
        password=request.password,
        email=request.email,
        role=request.role,
        created_by_user_id=actor.user_id,
        must_change_password=True,
    )


async def _guard_admin_transition(
    self,
    actor: UserActor,
    target: User,
    next_role: UserRole,
    next_status: UserStatus,
    active_admins: list[User],
) -> None:
    removes_active_admin = (
        target.role == "admin"
        and target.status == "active"
        and (next_role != "admin" or next_status != "active")
    )
    if actor.user_id == target.id and removes_active_admin:
        raise _error("USER_SELF_PROTECTED", "不能降级或禁用当前管理员。", 409)
    if removes_active_admin and len(active_admins) == 1:
        raise _error("USER_LAST_ADMIN_PROTECTED", "必须保留一个启用的管理员。", 409)
```

管理员进行 role/status 更新时先按 UUID 顺序锁定全部 active admin，再锁目标，保证并发请求的锁顺序一致。教师范围外目标统一 `USER_NOT_FOUND/404`。用户名和邮箱继续复用规范化、唯一性预检查及数据库唯一约束冲突映射。更新前保存旧 role/status；只有实际禁用或角色变化才撤销 Session。重置密码设置 `must_change_password=True` 并撤销全部 Session。

- [ ] **Step 6：增加 PostgreSQL 范围集成测试**

插入两个教师、各自学生和管理员：

```python
teacher_rows, teacher_total = await repository.list_managed(
    UserActor(teacher_a.id, "teacher"), query=None, role=None, status=None,
    page=1, page_size=20,
)
assert teacher_total == 1
assert [row[0].id for row in teacher_rows] == [student_a.id]

admin_rows, admin_total = await repository.list_managed(
    UserActor(admin.id, "admin"), query=None, role=None, status=None,
    page=1, page_size=20,
)
assert admin_total == 5
assert {row[0].id for row in admin_rows} == {
    teacher_a.id, teacher_b.id, student_a.id, student_b.id, admin.id
}
```

- [ ] **Step 7：验证并提交**

```powershell
docker compose run --rm app pytest -q tests/unit/auth/test_user_service.py tests/integration/test_user_repository.py
git add app/modules/users/schemas.py app/modules/users/repository.py app/modules/users/service.py tests/unit/auth/test_user_service.py tests/integration/test_user_repository.py
git commit -m "feat: enforce account management policy"
```

## Task 3：实现临时密码 principal 与自助修改密码

**Files:**
- Create: `app/modules/users/dependencies.py`
- Modify: `app/modules/auth/repository.py`
- Modify: `app/modules/auth/service.py`
- Modify: `app/modules/auth/schemas.py`
- Modify: `app/modules/auth/dependencies.py`
- Modify: `app/modules/auth/router.py`
- Modify: `app/modules/users/service.py`
- Modify: `app/modules/conversations/router.py`
- Modify: `app/modules/knowledge/router.py`
- Modify: `app/modules/rag/router.py`
- Modify: `tests/unit/auth/test_dependencies.py`
- Modify: `tests/unit/auth/test_user_service.py`
- Modify: `tests/integration/test_auth_sessions.py`
- Modify: `tests/api/test_auth.py`

- [ ] **Step 1：先写临时密码访问测试**

依赖测试：

```python
def test_password_ready_dependency_blocks_temporary_password() -> None:
    pending = principal(role="student", must_change_password=True)
    with pytest.raises(AppError) as exc_info:
        asyncio.run(require_password_ready(pending))
    assert exc_info.value.code == "AUTH_PASSWORD_CHANGE_REQUIRED"
    assert exc_info.value.status_code == 403
```

API 测试覆盖：临时密码登录后 `/auth/me` 返回 true；修改密码缺失 CSRF 为 403；错误当前密码为 422；正确修改为 204；旧 Session 失效；新密码重新登录后标记为 false。

- [ ] **Step 2：运行认证测试并确认失败**

```powershell
docker compose run --rm app pytest -q tests/unit/auth/test_dependencies.py tests/api/test_auth.py tests/integration/test_auth_sessions.py
```

Expected: FAIL，principal 无临时密码字段且修改密码端点不存在。

- [ ] **Step 3：把新角色和标记带入认证记录**

`LoginUserRecord`、`ActiveSessionRecord`、`AuthenticatedPrincipal`、`UserRead` 转换和 `AuthUserRead` 都增加标记，role 使用 `UserRole`：

```python
@dataclass(frozen=True)
class AuthenticatedPrincipal:
    user_id: UUID
    session_id: UUID
    username: str
    role: UserRole
    must_change_password: bool
    session_token_hash: bytes


class AuthUserRead(BaseModel):
    model_config = ConfigDict(from_attributes=True, frozen=True)
    id: UUID
    username: str
    email: str | None
    role: UserRole
    status: UserStatus
    must_change_password: bool
```

`AuthRepository.find_active_by_hash()` 和 `find_by_hash()` 每次从 User 行读取标记，不能使用登录时快照。

- [ ] **Step 4：装配事务级 UserService 并实现自助修改密码**

创建 `app/modules/users/dependencies.py`：

```python
from collections.abc import AsyncIterator


async def get_user_service() -> AsyncIterator[UserService]:
    session_factory = get_session_factory()
    async with session_factory() as session:
        async with session.begin():
            yield UserService(UserRepository(session), AuthRepository(session))
```

在 `UserService` 增加：

```python
async def change_own_password(
    self,
    user_id: UUID,
    current_password: str,
    new_password: str,
    now: datetime,
) -> None:
    _validate_password(new_password)
    user = await self._repository.get_by_id(user_id, for_update=True)
    if user is None or not await verify_password(current_password, user.password_hash):
        raise _error("AUTH_CURRENT_PASSWORD_INVALID", "当前密码不正确。", 422)
    if await verify_password(new_password, user.password_hash):
        raise _input_error("新密码不能与当前密码相同。")
    await self._repository.set_password_hash(
        user,
        await hash_password(new_password),
        now,
        must_change_password=False,
    )
    await self._revoke_sessions(user_id, now)
```

- [ ] **Step 5：增加请求模型和认证端点**

```python
class ChangePasswordRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    current_password: str = Field(min_length=1, max_length=128)
    new_password: str = Field(min_length=12, max_length=128)


@router.post("/change-password", status_code=status.HTTP_204_NO_CONTENT)
async def change_password(
    request: ChangePasswordRequest,
    principal: AuthenticatedPrincipal = Depends(require_csrf),
    service: UserService = Depends(get_user_service),
) -> Response:
    await service.change_own_password(
        principal.user_id,
        request.current_password,
        request.new_password,
        datetime.now(UTC),
    )
    return Response(status_code=status.HTTP_204_NO_CONTENT)
```

- [ ] **Step 6：增加“密码已就绪”依赖并替换业务端点依赖**

```python
async def require_password_ready(
    principal: AuthenticatedPrincipal = Depends(get_current_principal),
) -> AuthenticatedPrincipal:
    if principal.must_change_password:
        raise AppError(
            code="AUTH_PASSWORD_CHANGE_REQUIRED",
            message="请先修改临时密码。",
            status_code=403,
        )
    return principal


async def require_ready_csrf(
    principal: AuthenticatedPrincipal = Depends(require_csrf),
) -> AuthenticatedPrincipal:
    return await require_password_ready(principal)
```

`require_admin` 改依赖 `require_password_ready`，`require_admin_csrf` 改依赖 `require_ready_csrf`。会话和知识读取使用 `require_password_ready`；会话与 RAG 写入使用 `require_ready_csrf`。`/auth/me`、`/auth/logout`、`/auth/change-password` 仍允许临时密码账号访问。

- [ ] **Step 7：更新后端账号角色 fixture**

把以下文件中代表登录账号的 `user` 改为 `student`，不要修改 `messages.role='user'`：

```text
tests/unit/auth/test_dependencies.py
tests/api/test_auth.py
tests/api/test_knowledge_items.py
tests/api/test_documents.py
tests/api/test_conversations.py
tests/api/test_chat_v1.py
```

```powershell
rg -n 'principal\(role="user"\)|Literal\["admin", "user"\]' app tests
```

Expected: 无旧认证角色残留。

- [ ] **Step 8：验证并提交**

```powershell
docker compose run --rm app pytest -q tests/unit/auth tests/api/test_auth.py tests/api/test_conversations.py tests/api/test_chat_v1.py tests/api/test_knowledge_items.py tests/api/test_documents.py tests/integration/test_auth_sessions.py
git add app/modules/auth app/modules/users app/modules/conversations/router.py app/modules/knowledge/router.py app/modules/rag/router.py tests/unit/auth tests/api/test_auth.py tests/api/test_conversations.py tests/api/test_chat_v1.py tests/api/test_knowledge_items.py tests/api/test_documents.py tests/integration/test_auth_sessions.py
git commit -m "feat: require temporary password changes"
```

## Task 4：暴露用户管理 HTTP API

**Files:**
- Create: `app/modules/users/router.py`
- Create: `tests/api/test_users.py`
- Modify: `app/modules/auth/dependencies.py`
- Modify: `app/main.py`

- [ ] **Step 1：先写 HTTP 契约测试**

使用 fake service 和 dependency override 覆盖 OpenAPI、分页参数、学生 403、教师范围外 404、PATCH 空请求 422、CSRF/Origin 403、密码不泄露，并断言详情路由不存在 DELETE 操作。核心成功测试：

```python
def test_admin_and_teacher_routes_pass_actor_and_payload() -> None:
    client, service = build_client()
    created = client.post(
        "/api/v1/users",
        json={
            "username": "student-a",
            "email": "student-a@example.local",
            "password": "temporary-123",
            "role": "student",
        },
        headers=safe_headers(client, "teacher"),
    )
    listing = client.get(
        "/api/v1/users?q=student&role=student&status=active&page=2&page_size=10",
        headers={"X-Test-Role": "admin"},
    )
    assert created.status_code == 201
    assert created.json()["created_by_user_id"] == str(TEACHER_ID)
    assert listing.status_code == 200
    assert service.calls[0][0] == "create"
    assert service.calls[1][1]["page"] == 2
```

- [ ] **Step 2：确认路由测试失败**

```powershell
docker compose run --rm app pytest -q tests/api/test_users.py
```

Expected: FAIL，用户 router 不存在。

- [ ] **Step 3：增加用户管理依赖**

```python
async def require_user_manager(
    principal: AuthenticatedPrincipal = Depends(require_password_ready),
) -> AuthenticatedPrincipal:
    if principal.role not in {"teacher", "admin"}:
        raise AppError(code="AUTH_FORBIDDEN", message="权限不足。", status_code=403)
    return principal


async def require_user_manager_csrf(
    principal: AuthenticatedPrincipal = Depends(require_ready_csrf),
) -> AuthenticatedPrincipal:
    return await require_user_manager(principal)
```

- [ ] **Step 4：实现五个端点**

`app/modules/users/router.py` 使用 `UserActor(principal.user_id, principal.role)`：

```python
router = APIRouter(prefix="/api/v1/users", tags=["users"])


@router.get("", response_model=UserPage)
async def list_users(
    q: str | None = Query(default=None, min_length=1, max_length=320),
    role: UserRole | None = Query(default=None),
    status_filter: UserStatus | None = Query(default=None, alias="status"),
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    principal: AuthenticatedPrincipal = Depends(require_user_manager),
    service: UserService = Depends(get_user_service),
) -> UserPage:
    return await service.list_managed_users(
        UserActor(principal.user_id, principal.role),
        query=q, role=role, status=status_filter, page=page, page_size=page_size,
    )


@router.post("", response_model=ManagedUserRead, status_code=status.HTTP_201_CREATED)
async def create_user(
    request: UserCreate,
    principal: AuthenticatedPrincipal = Depends(require_user_manager_csrf),
    service: UserService = Depends(get_user_service),
) -> ManagedUserRead:
    return await service.create_managed_user(
        UserActor(principal.user_id, principal.role), request
    )
```

同文件实现 `GET /{user_id}`、`PATCH /{user_id}`、`POST /{user_id}/reset-password`；重置成功为 204，不返回密码。

- [ ] **Step 5：注册路由并验证**

```python
from app.modules.users.router import router as users_router

app.include_router(auth_router)
app.include_router(users_router)
```

```powershell
docker compose run --rm app pytest -q tests/api/test_users.py tests/api/test_auth.py tests/test_app_lifespan.py
```

Expected: PASS，OpenAPI 中用户路由只注册一次。

- [ ] **Step 6：提交**

```powershell
git add app/modules/auth/dependencies.py app/modules/users/router.py app/main.py tests/api/test_users.py
git commit -m "feat: expose user management api"
```

## Task 5：更新 CLI、OpenAPI 和后端兼容用例

**Files:**
- Modify: `scripts/create_user.py`
- Modify: `tests/integration/test_create_user_cli.py`
- Modify: `frontend/openapi.json`
- Modify: `frontend/src/api/schema.d.ts`
- Modify: tests containing legacy account-role fixtures

- [ ] **Step 1：先更新 CLI 测试**

```python
assert build_parser().parse_args(["--username", "student-a"]).role == "student"
assert build_parser().parse_args(
    ["--username", "teacher-a", "--role", "teacher"]
).role == "teacher"
assert user.created_by_user_id is None
assert user.must_change_password is True
```

- [ ] **Step 2：更新 CLI 角色并验证**

```python
parser.add_argument(
    "--role",
    choices=("student", "teacher", "admin"),
    default="student",
)
```

```powershell
docker compose run --rm app pytest -q tests/integration/test_create_user_cli.py
```

Expected: PASS。

- [ ] **Step 3：生成 OpenAPI 和 TypeScript 类型**

```powershell
python scripts/export_openapi.py
Set-Location frontend
npm run api:generate
npm run api:check
Set-Location ..
```

Expected: 生成契约包含三角色、`must_change_password`、用户 DTO 和五个管理端点。

- [ ] **Step 4：运行后端完整 runtime**

```powershell
docker compose run --rm app pytest -q -rs --ignore=tests/evaluation --ignore=tests/test_retrieval_baseline.py
```

Expected: PASS。若 `ck_users_role` 失败，只把代表账号角色的旧 `user` 改为 `student`；消息角色和 prompt 语义保持 `user`。

- [ ] **Step 5：检查并提交**

```powershell
git diff --cached --name-only
git add scripts/create_user.py tests/integration/test_create_user_cli.py frontend/openapi.json frontend/src/api/schema.d.ts
git commit -m "chore: update account roles and api schema"
```

暂存检查必须确认没有无关测试格式化或用户已有文件。

## Task 6：实现 Vue 临时密码修改闭环

**Files:**
- Create: `frontend/src/features/auth/ChangePasswordPage.vue`
- Modify: `frontend/src/features/auth/api.ts`
- Modify: `frontend/src/features/auth/useAuth.ts`
- Modify: `frontend/src/features/auth/LoginPage.vue`
- Modify: `frontend/src/features/auth/auth.spec.ts`
- Modify: `frontend/src/router/index.ts`
- Modify: `frontend/src/router/router.spec.ts`

- [ ] **Step 1：先写认证状态和守卫测试**

所有前端账号 fixture 改为 `role: 'student'` 并加入 `must_change_password: false`。新增：

```typescript
it('forces a temporary-password user to change password', async () => {
  const temporaryUser: AuthUser = {
    id: USER.id,
    username: USER.username,
    email: USER.email,
    role: USER.role,
    status: USER.status,
    must_change_password: true,
  }
  const { router } = await navigate(
    { status: 'authenticated', user: temporaryUser },
    '/chat',
  )
  expect(router.currentRoute.value.path).toBe('/change-password')
})

it('invalidates authentication after changing password', async () => {
  const api = createApi({ changePassword: vi.fn(async () => undefined) })
  const auth = createAuthController(api)
  await auth.login({ username: 'learner', password: 'temporary-123' })
  await auth.changePassword({
    current_password: 'temporary-123',
    new_password: 'permanent-456',
  })
  expect(auth.state.value.status).toBe('anonymous')
})
```

- [ ] **Step 2：确认测试失败**

```powershell
Set-Location frontend
npm test -- --run src/features/auth/auth.spec.ts src/router/router.spec.ts
Set-Location ..
```

Expected: FAIL，缺少 changePassword 和对应路由。

- [ ] **Step 3：扩展认证 API 和 controller**

```typescript
export type ChangePasswordRequest =
  components['schemas']['ChangePasswordRequest']

changePassword(values) {
  return apiRequest<void, ChangePasswordRequest>(
    '/api/v1/auth/change-password',
    { method: 'POST', body: values },
  )
}
```

`AuthController` 增加 `changePassword(values): Promise<void>`；成功后 `invalidate()`，失败时保持 authenticated 状态以便修正输入。

- [ ] **Step 4：创建修改密码页面**

页面使用当前密码、新密码、确认密码三个字段，校验新密码 12～128 位且两次一致：

```vue
<form class="password-form" novalidate @submit.prevent="submit">
  <label for="current-password">当前密码</label>
  <input id="current-password" v-model="currentPassword" type="password" autocomplete="current-password" />
  <label for="new-password">新密码</label>
  <input id="new-password" v-model="newPassword" type="password" autocomplete="new-password" />
  <label for="confirm-password">确认新密码</label>
  <input id="confirm-password" v-model="confirmation" type="password" autocomplete="new-password" />
  <InlineAlert v-if="error" tone="error" title="密码修改失败">
    <p>{{ error.message }}</p>
  </InlineAlert>
  <button type="submit" :disabled="submitting">
    {{ submitting ? '正在修改' : '修改密码' }}
  </button>
</form>
```

成功路径：

```typescript
await auth.changePassword({
  current_password: currentPassword.value,
  new_password: newPassword.value,
})
await router.replace({ name: 'login', query: { password_changed: '1' } })
```

`LoginPage.vue` 对该 query 显示“密码已修改，请重新登录。”。

- [ ] **Step 5：加入路由和强制守卫**

```typescript
{
  path: '/change-password',
  name: 'change-password',
  component: () => import('../features/auth/ChangePasswordPage.vue'),
  meta: { requiresAuth: true, title: '修改密码' },
},

if (
  state.status === 'authenticated' &&
  state.user.must_change_password &&
  to.name !== 'change-password'
) return { name: 'change-password' }

if (
  state.status === 'authenticated' &&
  !state.user.must_change_password &&
  to.name === 'change-password'
) return '/chat'
```

强制密码判断位于 bootstrap 之后、普通业务权限判断之前。

- [ ] **Step 6：验证并提交**

```powershell
Set-Location frontend
npm test -- --run src/features/auth/auth.spec.ts src/router/router.spec.ts
npm run typecheck
Set-Location ..
git add frontend/src/features/auth frontend/src/router
git commit -m "feat: add forced password change ui"
```

## Task 7：实现完整用户管理界面

**Files:**
- Create: `frontend/src/features/users/types.ts`
- Create: `frontend/src/features/users/api.ts`
- Create: `frontend/src/features/users/useUsers.ts`
- Create: `frontend/src/features/users/UserListPage.vue`
- Create: `frontend/src/features/users/UserEditorPage.vue`
- Create: `frontend/src/features/users/ResetPasswordDialog.vue`
- Create: `frontend/src/features/users/users.spec.ts`
- Modify: `frontend/src/router/index.ts`
- Modify: `frontend/src/router/meta.d.ts`
- Modify: `frontend/src/router/router.spec.ts`
- Modify: `frontend/src/app/AppNavigation.vue`
- Modify: `frontend/src/app/AppShell.spec.ts`

- [ ] **Step 1：先写列表和表单测试**

`users.spec.ts` 至少覆盖教师固定学生角色、管理员筛选三角色、状态确认和重置密码：

```typescript
it('teacher creation form fixes the role to student', async () => {
  await renderEditor(TEACHER, '/users/new')
  expect(screen.queryByLabelText('角色')).toBeNull()
  await fireEvent.update(screen.getByLabelText('用户名'), 'student-a')
  await fireEvent.update(screen.getByLabelText('临时密码'), 'temporary-123')
  await fireEvent.update(screen.getByLabelText('确认临时密码'), 'temporary-123')
  await fireEvent.click(screen.getByRole('button', { name: '创建账号' }))
  expect(usersApi.create).toHaveBeenCalledWith(
    expect.objectContaining({ role: 'student', username: 'student-a' }),
  )
})

it('requires confirmation before disabling an account', async () => {
  await renderEditor(ADMIN, `/users/${STUDENT.id}`)
  await screen.findByDisplayValue('student-a')
  await fireEvent.click(screen.getByRole('switch', { name: '账号已启用' }))
  expect(screen.getByRole('alertdialog', { name: '禁用账号' })).toBeTruthy()
})
```

再覆盖：加载、空列表、错误、分页、角色/状态/关键词筛选、冲突错误、管理员三角色选择、密码不一致、重置成功和教师无角色控件。

- [ ] **Step 2：确认模块缺失**

```powershell
Set-Location frontend
npm test -- --run src/features/users/users.spec.ts src/app/AppShell.spec.ts src/router/router.spec.ts
Set-Location ..
```

Expected: FAIL，users feature 尚不存在。

- [ ] **Step 3：定义类型和 API client**

```typescript
export type ManagedUser = components['schemas']['ManagedUserRead']
export type UserPage = components['schemas']['UserPage']
export type UserCreate = components['schemas']['UserCreate']
export type UserUpdate = components['schemas']['UserUpdate']
export type UserPasswordReset = components['schemas']['UserPasswordReset']
export type UserRole = ManagedUser['role']
export type UserStatus = ManagedUser['status']

export interface UserFilters {
  query?: string
  role?: UserRole
  status?: UserStatus
  page: number
  pageSize: number
}
```

`usersApi` 精确实现 list/get/create/update/resetPassword：

```typescript
list(filters, signal) {
  const params = new URLSearchParams({
    page: String(filters.page),
    page_size: String(filters.pageSize),
  })
  if (filters.query) params.set('q', filters.query)
  if (filters.role) params.set('role', filters.role)
  if (filters.status) params.set('status', filters.status)
  return apiRequest<UserPage>(`/api/v1/users?${params.toString()}`, { signal })
},
resetPassword(id, values) {
  return apiRequest<void, UserPasswordReset>(
    `/api/v1/users/${encodeURIComponent(id)}/reset-password`,
    { method: 'POST', body: values },
  )
}
```

- [ ] **Step 4：实现可取消列表状态**

`useUsers.ts` 沿用知识列表的 last-request-wins 模式：每次 load 取消前一个 `AbortController`，`AbortError` 不覆盖新请求，refresh 使用最后筛选。状态固定为：

```typescript
export type UserListState =
  | { status: 'idle'; data: null; error: null }
  | { status: 'loading'; data: UserPage | null; error: null }
  | { status: 'ready'; data: UserPage; error: null }
  | { status: 'error'; data: UserPage | null; error: ApiError }
```

- [ ] **Step 5：实现账号列表**

URL query 保存 `q/role/status/page`。管理员显示角色筛选，教师不显示；桌面列为用户名、邮箱、角色、状态、创建者、创建时间、操作：

```vue
<form class="user-filters" @submit.prevent="applyFilters">
  <label>搜索<input v-model="queryInput" type="search" /></label>
  <label v-if="isAdmin">角色<select :value="role ?? ''" @change="setRole">
    <option value="">全部角色</option>
    <option value="student">学生</option>
    <option value="teacher">教师</option>
    <option value="admin">管理员</option>
  </select></label>
  <label>状态<select :value="status ?? ''" @change="setStatus">
    <option value="">全部状态</option>
    <option value="active">启用</option>
    <option value="disabled">禁用</option>
  </select></label>
  <button type="submit">应用筛选</button>
</form>
<RouterLink class="primary-command" to="/users/new">创建账号</RouterLink>
```

移动端将 table row 转成带 `data-label` 的紧凑分行布局，页面不得横向溢出。

- [ ] **Step 6：实现创建/编辑和密码重置**

`UserEditorPage.vue` 以 `route.name === 'user-new'` 区分模式。教师创建始终提交 `role: 'student'`；管理员显示角色 select。详情模式只 PATCH 变化字段，状态 switch 先打开确认框。管理员编辑自己时禁用角色与状态控件，后端继续作最终保护。

`ResetPasswordDialog.vue` 要求输入并确认临时密码：

```vue
<ConfirmDialog
  :open="open"
  :busy="submitting"
  title="重置临时密码"
  :object-name="username"
  confirm-label="确认重置"
  @cancel="$emit('cancel')"
  @confirm="submit"
>
  <label for="reset-password">临时密码</label>
  <input id="reset-password" v-model="password" type="password" autocomplete="new-password" />
  <label for="reset-confirmation">确认临时密码</label>
  <input id="reset-confirmation" v-model="confirmation" type="password" autocomplete="new-password" />
</ConfirmDialog>
```

- [ ] **Step 7：加入路由、守卫和导航**

`meta.d.ts` 增加 `requiresUserManager?: boolean`。加入 `/users`、`/users/new`、`/users/:id`，均 requiresAuth + requiresUserManager。守卫：

```typescript
if (
  to.meta.requiresUserManager &&
  (state.status !== 'authenticated' ||
    !['teacher', 'admin'].includes(state.user.role))
) return '/chat'
```

`AppNavigation.vue` 引入 Lucide `Users` 图标。管理区对 teacher/admin 显示“用户管理”；知识库、文档和摄取任务继续只对 admin 显示。

- [ ] **Step 8：运行质量检查并提交**

```powershell
Set-Location frontend
npm test -- --run src/features/users/users.spec.ts src/app/AppShell.spec.ts src/router/router.spec.ts
npm run typecheck
npm run lint
npm run format:check
Set-Location ..
git add frontend/src/features/users frontend/src/router frontend/src/app/AppNavigation.vue frontend/src/app/AppShell.spec.ts
git commit -m "feat: add user management interface"
```

Expected: 全部 PASS。

## Task 8：端到端流程、文档与完整验收

**Files:**
- Create: `frontend/tests/e2e/account-management.spec.ts`
- Modify: `frontend/tests/e2e/fixtures.ts`
- Modify: `README.md`

- [ ] **Step 1：扩展 Playwright mock**

`MockApiState` 增加：

```typescript
role: 'admin' | 'teacher' | 'student'
mustChangePassword: boolean
users: ManagedUserFixture[]
```

`/auth/me` 返回 `must_change_password`；实现 `/auth/change-password` 及用户 list/create/get/patch/reset-password。教师 list 固定过滤 `role === 'student' && created_by_user_id === IDS.teacher`。

- [ ] **Step 2：写四条 E2E**

```typescript
test('administrator creates a teacher', async ({ page }) => {
  await installMockApi(page, { role: 'admin' })
  await page.goto('/users/new')
  await page.getByLabel('用户名').fill('teacher-a')
  await page.getByLabel('角色').selectOption('teacher')
  await page.getByLabel('临时密码').fill('temporary-123')
  await page.getByLabel('确认临时密码').fill('temporary-123')
  await page.getByRole('button', { name: '创建账号' }).click()
  await expect(page.getByDisplayValue('teacher-a')).toBeVisible()
})

test('teacher sees only owned students', async ({ page }) => {
  await installMockApi(page, { role: 'teacher' })
  await page.goto('/users')
  await expect(page.getByRole('link', { name: 'owned-student' })).toBeVisible()
  await expect(page.getByText('other-student')).toHaveCount(0)
})

test('student cannot open user management', async ({ page }) => {
  await installMockApi(page, { role: 'student' })
  await page.goto('/users')
  await expect(page).toHaveURL('/chat')
  await expect(page.getByRole('link', { name: '用户管理' })).toHaveCount(0)
})

test('temporary password must be changed before work', async ({ page }) => {
  await installMockApi(page, { role: 'teacher', mustChangePassword: true })
  await page.goto('/chat')
  await expect(page).toHaveURL('/change-password')
  await page.getByLabel('当前密码').fill('temporary-123')
  await page.getByLabel('新密码').fill('permanent-456')
  await page.getByLabel('确认新密码').fill('permanent-456')
  await page.getByRole('button', { name: '修改密码' }).click()
  await expect(page).toHaveURL(/\/login\?password_changed=1/)
})
```

- [ ] **Step 3：运行完整前端质量链**

```powershell
Set-Location frontend
npm run api:check
npm run format:check
npm run lint
npm run typecheck
npm test -- --run
npm run build
npm run e2e
Set-Location ..
```

Expected: 全部 PASS，无文本溢出、重叠或不可访问控件。

- [ ] **Step 4：更新 README**

在账号创建章节加入：

```markdown
### 账号与角色

- 管理员可以创建和管理学生、教师、管理员。
- 教师只能创建并管理自己创建的学生。
- 学生没有账号管理入口。
- 新建账号和重置密码后必须在首次登录时修改临时密码。
- 系统不提供公开注册和账号物理删除。

```powershell
python scripts/create_user.py --username admin --role admin --email admin@example.local
python scripts/create_user.py --username teacher01 --role teacher --email teacher01@example.local
```
```

- [ ] **Step 5：运行后端 runtime、Compose 契约和迁移烟测**

```powershell
docker compose run --rm app pytest -q -rs --ignore=tests/evaluation --ignore=tests/test_retrieval_baseline.py
docker compose run --rm app pytest -q tests/test_compose_contract.py
docker compose up -d --build mathrag postgres
docker compose exec -T mathrag alembic current
docker compose ps
```

Expected: 测试全部 PASS；Alembic 为 `0006_add_account_management (head)`；应用与 PostgreSQL healthy。实际调用 login、users、change-password，响应不得出现 `password` 或 `password_hash`。

- [ ] **Step 6：检查差异并提交**

```powershell
git diff --check
git status --short
git add README.md frontend/tests/e2e/fixtures.ts frontend/tests/e2e/account-management.spec.ts
git commit -m "test: verify account management workflows"
```

不得暂存用户已有的 `docs/superpowers/plans/2026-07-31-mathrag-figma-student-ui.md` 或 `tmp/`。

## 最终完成检查

- [ ] 旧 `user` 已迁移为 `student`，旧 `admin` 保持不变。
- [ ] 创建者自关联、临时密码字段、外键、索引和降级保护已验证。
- [ ] 管理员可管理三角色，但不能降级或禁用自己及最后一个启用管理员。
- [ ] 教师只能完整管理自己创建的学生；管理员创建的学生不属于教师。
- [ ] 教师禁用或降级后归属保留，恢复后可继续管理原学生。
- [ ] 学生无法访问用户管理 API 或页面。
- [ ] 新建账号与重置密码设置临时密码标记并撤销旧 Session。
- [ ] 修改密码后撤销 Session，用户必须用新密码重新登录。
- [ ] 教师没有知识库、文档和摄取任务管理权限。
- [ ] 不存在公开注册、物理删除、学生转移或教师读取学生会话功能。
- [ ] OpenAPI、TypeScript 类型、CLI 和 README 已同步。
- [ ] 后端 runtime、前端质量链、Playwright 与 Compose 契约全部通过。
