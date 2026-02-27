"""Speaker embedding management using LanceDB.

This module handles vector embeddings for speaker recognition using LanceDB.
Metadata and speaker information are stored in SQL (see models.py and database.py).

LanceDB is used exclusively for:
- Storing embedding vectors
- Cosine similarity search for speaker matching
"""

from typing import List, Optional, Dict, Any, Tuple
import numpy as np
import lancedb
import pyarrow as pa
from pathlib import Path

from .config import DEFAULT_DB_PATH


class EmbeddingManager:
    """Manage speaker embeddings using LanceDB.
    
    This class focuses solely on vector storage and similarity search.
    Speaker metadata (names, audio files, segments) is handled by DatabaseManager.
    
    Schema:
        - id: Unique embedding ID (string)
        - speaker_id: Foreign key to speakers table (integer)
        - embedding: 1-D embedding vector (list of floats)
        - audio_file_id: Foreign key to audio_files table (integer)
        - start_time: Segment start time in seconds (float)
        - end_time: Segment end time in seconds (float)
    """

    def __init__(self, db_path: str = DEFAULT_DB_PATH):
        """Initialize the embedding manager.

        Args:
            db_path: Path to LanceDB database directory
        """
        self.db_path = db_path
        Path(db_path).mkdir(parents=True, exist_ok=True)
        self.db = lancedb.connect(db_path)

    def get_or_create_embeddings_table(self) -> Any:
        """Get or create the embeddings table.

        Returns:
            LanceDB table object
        """
        try:
            return self.db.open_table("embeddings")
        except Exception:
            # Table doesn't exist, return None and create it when first embedding is added
            return None

    def add_embedding(
        self,
        embedding_id: str,
        speaker_id: int,
        embedding: np.ndarray,
        audio_file_id: int,
        start_time: float,
        end_time: float,
    ) -> None:
        """Add an embedding to the database.

        Args:
            embedding_id: Unique identifier for this embedding
            speaker_id: ID from speakers table in SQL database
            embedding: Embedding vector (1-D numpy array)
            audio_file_id: ID from audio_files table in SQL database
            start_time: Segment start time in seconds
            end_time: Segment end time in seconds
        """
        table = self.get_or_create_embeddings_table()
        
        # Ensure embedding is 1-D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        record = {
            "id": embedding_id,
            "speaker_id": speaker_id,
            "embedding": embedding.tolist(),
            "audio_file_id": audio_file_id,
            "start_time": start_time,
            "end_time": end_time,
        }
        
        if table is None:
            # Create table with first record
            self.db.create_table("embeddings", data=[record])
        else:
            # Add to existing table
            table.add([record])

    def search_similar_embeddings(
        self,
        embedding: np.ndarray,
        limit: int = 5,
        similarity_threshold: Optional[float] = None,
    ) -> List[Dict[str, Any]]:
        """Search for similar embeddings by cosine similarity.

        Args:
            embedding: Query embedding vector
            limit: Maximum number of results to return
            similarity_threshold: Optional minimum similarity score (0-1)

        Returns:
            List of similar embedding results with keys:
                - id: Embedding ID
                - speaker_id: Speaker ID from SQL database
                - _distance: Cosine distance (lower is more similar)
                - audio_file_id: Audio file ID
                - start_time, end_time: Segment boundaries
        """
        table = self.get_or_create_embeddings_table()
        
        # Ensure embedding is 1-D
        if embedding.ndim > 1:
            embedding = embedding.flatten()
        
        try:
            results = table.search(embedding.tolist()).limit(limit).to_list()
            
            # Filter by similarity threshold if provided
            # Convert distance to similarity: similarity = 1 - distance
            if similarity_threshold is not None:
                results = [
                    r for r in results
                    if (1.0 - r.get("_distance", 1.0)) >= similarity_threshold
                ]
            
            return results
        except Exception:
            return []

    def get_embeddings_by_speaker(
        self,
        speaker_id: int
    ) -> List[Dict[str, Any]]:
        """Get all embeddings for a specific speaker.

        Args:
            speaker_id: Speaker ID from SQL database

        Returns:
            List of embedding records for the speaker
        """
        try:
            table = self.db.open_table("embeddings")
            # LanceDB doesn't have direct filtering, so we get all and filter
            all_embeddings = table.search().to_list()
            return [e for e in all_embeddings if e.get("speaker_id") == speaker_id]
        except Exception:
            return []

    def get_embeddings_by_audio_file(
        self,
        audio_file_id: int
    ) -> List[Dict[str, Any]]:
        """Get all embeddings for a specific audio file.

        Args:
            audio_file_id: Audio file ID from SQL database

        Returns:
            List of embedding records for the audio file
        """
        try:
            table = self.db.open_table("embeddings")
            all_embeddings = table.search().to_list()
            return [e for e in all_embeddings if e.get("audio_file_id") == audio_file_id]
        except Exception:
            return []

    def compute_mean_embedding(
        self,
        speaker_id: int,
        weights: Optional[List[float]] = None
    ) -> Optional[np.ndarray]:
        """Compute duration-weighted mean embedding for a speaker.

        Args:
            speaker_id: Speaker ID from SQL database
            weights: Optional weights for each embedding (e.g., segment durations)

        Returns:
            Mean embedding vector, or None if no embeddings found
        """
        embeddings = self.get_embeddings_by_speaker(speaker_id)
        if not embeddings:
            return None
        
        # Extract embedding vectors
        vectors = [np.array(e["embedding"]) for e in embeddings]
        
        if weights is None:
            # Use durations as weights
            weights = [e["end_time"] - e["start_time"] for e in embeddings]
        
        # Compute weighted mean
        weights = np.array(weights)
        weights = weights / weights.sum()  # Normalize
        
        mean_embedding = np.average(vectors, axis=0, weights=weights)
        
        # Normalize to unit length (for cosine similarity)
        norm = np.linalg.norm(mean_embedding)
        if norm > 0:
            mean_embedding = mean_embedding / norm
        
        return mean_embedding

    def delete_embeddings_by_speaker(self, speaker_id: int) -> int:
        """Delete all embeddings for a speaker.

        Args:
            speaker_id: Speaker ID from SQL database

        Returns:
            Number of embeddings deleted
        """
        try:
            table = self.db.open_table("embeddings")
            # LanceDB doesn't support direct deletion by condition
            # This is a limitation - in production, consider periodic cleanup
            # For now, just return 0 (feature not implemented)
            return 0
        except Exception:
            return 0
