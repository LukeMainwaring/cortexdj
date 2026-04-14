"""add pgvector extension and track_audio_embeddings table

Revision ID: 77c744e4b096
Revises: 36df67a68339
Create Date: 2026-04-14 15:07:58.198246

"""

import pgvector.sqlalchemy
import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision = "77c744e4b096"
down_revision = "36df67a68339"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    op.create_table(
        "track_audio_embeddings",
        sa.Column("id", sa.Integer(), autoincrement=True, nullable=False),
        sa.Column("spotify_id", sa.String(length=64), nullable=False),
        sa.Column("itunes_track_id", sa.String(length=64), nullable=True),
        sa.Column("itunes_preview_url", sa.String(length=1024), nullable=True),
        sa.Column("title", sa.String(length=512), nullable=False),
        sa.Column("artist", sa.String(length=512), nullable=False),
        sa.Column("source", sa.String(length=32), nullable=False),
        sa.Column("embedding", pgvector.sqlalchemy.Vector(512), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_track_audio_embeddings_spotify_id"),
        "track_audio_embeddings",
        ["spotify_id"],
        unique=True,
    )
    # HNSW over IVFFlat: no training step (IVFFlat would be built on an empty
    # table here and degrade to a scan), handles incremental seed inserts
    # without reindexing, better recall at our ~2k-10k row scale.
    op.execute(
        """
        CREATE INDEX ix_track_audio_embeddings_embedding_hnsw
        ON track_audio_embeddings USING hnsw (embedding vector_cosine_ops)
        WITH (m = 16, ef_construction = 64)
        """
    )


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS ix_track_audio_embeddings_embedding_hnsw")
    op.drop_index(op.f("ix_track_audio_embeddings_spotify_id"), table_name="track_audio_embeddings")
    op.drop_table("track_audio_embeddings")
