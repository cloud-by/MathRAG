"""新增账号角色、创建归属与临时密码状态。"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "0006_add_account_management"
down_revision: str | None = "0005_create_documents_ingestion_jobs"
branch_labels: str | None = None
depends_on: str | None = None


def upgrade() -> None:
    """迁移旧角色并新增账号管理字段。"""
    op.add_column(
        "users",
        sa.Column(
            "created_by_user_id",
            postgresql.UUID(as_uuid=True),
            nullable=True,
        ),
    )
    op.add_column(
        "users",
        sa.Column(
            "must_change_password",
            sa.Boolean(),
            server_default=sa.false(),
            nullable=False,
        ),
    )

    # 先移除旧约束，避免旧角色集合阻止数据转换。
    op.drop_constraint(op.f("ck_users_role"), "users", type_="check")
    op.execute(sa.text("UPDATE users SET role = 'student' WHERE role = 'user'"))
    op.create_check_constraint(
        op.f("ck_users_role"),
        "users",
        "role IN ('student', 'teacher', 'admin')",
    )
    op.alter_column("users", "role", server_default="student")

    op.create_foreign_key(
        "fk_users_created_by_user_id_users",
        "users",
        "users",
        ["created_by_user_id"],
        ["id"],
        ondelete="SET NULL",
    )
    op.create_index(
        "ix_users_created_by_user_id",
        "users",
        ["created_by_user_id"],
    )


def downgrade() -> None:
    """无 teacher 账号时恢复旧角色与模式。"""
    teacher_count = op.get_bind().scalar(
        sa.text("SELECT count(*) FROM users WHERE role = 'teacher'")
    )
    if teacher_count:
        raise RuntimeError("存在 teacher 账号，无法降级账号管理迁移。")

    op.drop_index("ix_users_created_by_user_id", table_name="users")
    op.drop_constraint(
        "fk_users_created_by_user_id_users",
        "users",
        type_="foreignkey",
    )

    # 先移除新约束，避免新角色集合阻止数据转换。
    op.drop_constraint(op.f("ck_users_role"), "users", type_="check")
    op.execute(sa.text("UPDATE users SET role = 'user' WHERE role = 'student'"))
    op.create_check_constraint(
        op.f("ck_users_role"),
        "users",
        "role IN ('admin', 'user')",
    )
    op.alter_column("users", "role", server_default="user")

    op.drop_column("users", "must_change_password")
    op.drop_column("users", "created_by_user_id")
