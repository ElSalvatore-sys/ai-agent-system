"""Add local-LLM support

Revision ID: 002_local_llm_support
Revises: 001_comprehensive_schema
Create Date: 2024-08-01 00:00:00.000000
"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '002_local_llm_support'
down_revision = '001_comprehensive_schema'
branch_labels = None
depends_on = None


# NOTE: SQLite has limited ALTER TABLE; alembic autogenerate for enums doesn't
# work.  We therefore add new values to the `modelprovider` enum *only* when the
# backend runs against PostgreSQL.  For SQLite the enum is just a CHECK
# constraint so we can safely skip it.

def _add_values_to_modelprovider_enum():
    bind = op.get_bind()
    if bind.dialect.name != 'postgresql':
        return  # nothing to do on SQLite / MySQL etc.

    # Add enum values
    op.execute("""
        ALTER TYPE modelprovider ADD VALUE IF NOT EXISTS 'local_ollama';
    """)
    op.execute("""
        ALTER TYPE modelprovider ADD VALUE IF NOT EXISTS 'local_hf';
    """)


def upgrade():
    # 1) Extend enum (PostgreSQL only)
    _add_values_to_modelprovider_enum()

    # 2) New columns on ai_models
    with op.batch_alter_table('ai_models') as batch_op:
        batch_op.add_column(sa.Column('host_type', sa.String(length=20), nullable=True, server_default='cloud'))
        batch_op.add_column(sa.Column('device_id', sa.String(length=64), nullable=True))
        batch_op.add_column(sa.Column('availability', sa.Boolean(), nullable=True, server_default=sa.text('true')))


def downgrade():
    # Remove columns
    with op.batch_alter_table('ai_models') as batch_op:
        batch_op.drop_column('availability')
        batch_op.drop_column('device_id')
        batch_op.drop_column('host_type')

    # Enum downgrade left as-is (safe no-op).
