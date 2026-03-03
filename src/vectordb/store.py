"""Qdrant vector store for document region embeddings."""

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams


@dataclass
class RegionMetadata:
    """Metadata stored alongside each region embedding."""

    doc_name: str
    page_index: int
    region_index: int
    label: str
    bbox: list[int]
    image_path: str


class VectorStore:
    """Qdrant-backed vector store for region embeddings."""

    def __init__(
        self,
        host: str = "localhost",
        port: int = 6333,
        collection_name: str = "document_regions",
    ):
        self.client = QdrantClient(host=host, port=port)
        self.collection_name = collection_name

    def create_collection(self, embedding_dim: int) -> None:
        """Create or recreate the collection."""
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=embedding_dim,
                distance=Distance.COSINE,
            ),
        )

    def upsert_regions(
        self,
        embeddings: np.ndarray,
        metadata_list: list[RegionMetadata],
        start_id: int = 0,
    ) -> None:
        """Insert region embeddings with metadata into Qdrant.

        Args:
            embeddings: Array of shape (N, D).
            metadata_list: Metadata for each embedding.
            start_id: Starting point ID.
        """
        points = []
        for i, (emb, meta) in enumerate(zip(embeddings, metadata_list)):
            points.append(
                PointStruct(
                    id=start_id + i,
                    vector=emb.tolist(),
                    payload=asdict(meta),
                )
            )
        # Upsert in batches of 100
        for i in range(0, len(points), 100):
            self.client.upsert(
                collection_name=self.collection_name,
                points=points[i : i + 100],
            )

    def search(
        self,
        query_vector: np.ndarray | list[float],
        top_k: int = 5,
    ) -> list[dict[str, Any]]:
        """Search for the top-K most similar regions.

        Args:
            query_vector: Query embedding vector.
            top_k: Number of results to return.

        Returns:
            List of dicts with keys: id, score, and all metadata fields.
        """
        if isinstance(query_vector, np.ndarray):
            query_vector = query_vector.tolist()

        results = self.client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=top_k,
        )
        return [
            {"id": hit.id, "score": hit.score, **hit.payload}
            for hit in results
        ]

    def count(self) -> int:
        """Return the number of vectors in the collection."""
        info = self.client.get_collection(self.collection_name)
        return info.points_count
