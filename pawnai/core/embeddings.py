"""Speaker embedding management using LanceDB."""

from typing import List, Optional, Dict, Any
import numpy as np
import lancedb
import pyarrow as pa
from pathlib import Path

from .config import DEFAULT_DB_PATH


class EmbeddingManager:
    """Manage speaker embeddings using LanceDB."""

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the embedding manager.

        Args:
            db_path: Path to LanceDB database directory
        """
        self.db_path = db_path
        self.db = lancedb.connect(db_path)

    def get_or_create_speaker_names_table(self) -> Any:
        """Get or create the speaker_names table.

        Returns:
            LanceDB table object
        """
        try:
            return self.db.open_table("speaker_names")
        except Exception:
            # Create table with initial schema if it doesn't exist
            return self.db.create_table(
                "speaker_names",
                data=pa.table(
                    {
                        "speaker_id": pa.array([], type=pa.string()),
                        "name": pa.array([], type=pa.string()),
                        "count": pa.array([], type=pa.int32()),
                    }
                ),
                mode="overwrite",
            )

    def get_or_create_embeddings_table(self) -> Any:
        """Get or create the embeddings table.

        Returns:
            LanceDB table object
        """
        try:
            return self.db.open_table("embeddings")
        except Exception:
            # Create table with initial schema if it doesn't exist
            return self.db.create_table(
                "embeddings",
                data=pa.table(
                    {
                        "speaker_id": pa.array([], type=pa.string()),
                        "embedding": pa.array([], type=pa.list_(pa.float32())),
                        "audio_path": pa.array([], type=pa.string()),
                    }
                ),
                mode="overwrite",
            )

    def add_embedding(
        self, speaker_id: str, embedding: np.ndarray, audio_path: str
    ) -> None:
        """Add an embedding to the database.

        Args:
            speaker_id: Unique speaker identifier
            embedding: Embedding vector
            audio_path: Path to the audio file
        """
        table = self.get_or_create_embeddings_table()
        table.add(
            [
                {
                    "speaker_id": speaker_id,
                    "embedding": embedding.tolist(),
                    "audio_path": audio_path,
                }
            ]
        )

    def add_speaker_name(
        self, speaker_id: str, name: str, count: int = 1
    ) -> None:
        """Add or update a speaker name.

        Args:
            speaker_id: Unique speaker identifier
            name: Human-readable speaker name
            count: Number of samples
        """
        table = self.get_or_create_speaker_names_table()
        table.add([{"speaker_id": speaker_id, "name": name, "count": count}])

    def search_similar_speakers(
        self, embedding: np.ndarray, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """Search for similar speakers by embedding.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results

        Returns:
            List of similar speaker results
        """
        table = self.get_or_create_embeddings_table()
        results = table.search(embedding.tolist()).limit(limit).to_list()
        return results

    def get_all_embeddings(self) -> List[Dict[str, Any]]:
        """Retrieve all embeddings from the database.

        Returns:
            List of embedding records
        """
        try:
            table = self.db.open_table("embeddings")
            return table.search().to_list()
        except Exception:
            return []

    def get_speaker_names(self) -> Dict[str, str]:
        """Get all speaker names.

        Returns:
            Dictionary mapping speaker_id to name
        """
        try:
            table = self.db.open_table("speaker_names")
            records = table.search().to_list()
            return {r["speaker_id"]: r["name"] for r in records}
        except Exception:
            return {}
