"""
src/rag/retriever.py
─────────────────────────────────────────────────────────────
Semantic vector retriever backed by ChromaDB + Granite Embeddings.

Provides two retrievers:
  - FoodRetriever: query the USDA foods collection
  - GuidelineRetriever: query the disease guidelines collection
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.services.watsonx_service import get_embedding_client

logger = logging.getLogger(__name__)


@dataclass
class RetrievedChunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0  # cosine similarity (1 = most similar)


class _EmbeddingFunction(chromadb.EmbeddingFunction):
    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:  # noqa: A002
        return get_embedding_client().embed_documents(list(input))


class _BaseRetriever:
    def __init__(self, collection_name: str) -> None:
        cfg = get_settings()
        Path(cfg.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(
            path=cfg.chroma_persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name=collection_name,
            embedding_function=_EmbeddingFunction(),
            metadata={"hnsw:space": "cosine"},
        )

    def query(
        self,
        query_text: str,
        n_results: int = 5,
        where: dict | None = None,
    ) -> list[RetrievedChunk]:
        """
        Perform semantic search and return ranked chunks.

        Args:
            query_text: Natural language query.
            n_results: Maximum number of results.
            where: Optional ChromaDB metadata filter dict.

        Returns:
            List of RetrievedChunk sorted by similarity (descending).
        """
        try:
            kwargs: dict[str, Any] = {
                "query_texts": [query_text],
                "n_results": n_results,
                "include": ["documents", "metadatas", "distances"],
            }
            if where:
                kwargs["where"] = where

            results = self._collection.query(**kwargs)

            chunks: list[RetrievedChunk] = []
            docs = results.get("documents", [[]])[0]
            metas = results.get("metadatas", [[]])[0]
            distances = results.get("distances", [[]])[0]

            for doc, meta, dist in zip(docs, metas, distances):
                # ChromaDB returns L2 distance for cosine; convert to similarity
                similarity = 1.0 - dist
                chunks.append(
                    RetrievedChunk(
                        id=meta.get("food_id") or meta.get("condition_key", ""),
                        text=doc,
                        metadata=meta,
                        score=similarity,
                    )
                )
            return chunks

        except Exception as exc:
            logger.error("Retrieval failed for query '%s': %s", query_text, exc)
            return []


class FoodRetriever(_BaseRetriever):
    """
    Retrieves food nutritional data from the USDA foods collection.

    Example:
        retriever = FoodRetriever()
        results = retriever.query("high protein low carb breakfast foods")
    """

    def __init__(self) -> None:
        super().__init__(get_settings().chroma_collection_foods)

    def get_food_by_name(self, food_name: str) -> RetrievedChunk | None:
        """Exact-name lookup with semantic fallback."""
        results = self.query(food_name, n_results=1)
        return results[0] if results else None

    def filter_by_allergen_free(
        self, query_text: str, excluded_allergens: list[str], n_results: int = 5
    ) -> list[RetrievedChunk]:
        """
        Retrieve foods avoiding specific allergens.
        Post-filters since ChromaDB doesn't support 'not contains' on JSON arrays.
        """
        candidates = self.query(query_text, n_results=n_results * 3)
        filtered = []
        for chunk in candidates:
            allergens_raw = chunk.metadata.get("allergens", "[]")
            allergens: list[str] = json.loads(allergens_raw) if isinstance(allergens_raw, str) else allergens_raw
            if not any(a.lower() in [x.lower() for x in excluded_allergens] for a in allergens):
                filtered.append(chunk)
            if len(filtered) >= n_results:
                break
        return filtered


class GuidelineRetriever(_BaseRetriever):
    """
    Retrieves clinical dietary guidelines from the disease guidelines collection.

    Example:
        retriever = GuidelineRetriever()
        results = retriever.query("low glycemic diet for diabetes")
    """

    def __init__(self) -> None:
        super().__init__(get_settings().chroma_collection_guidelines)

    def get_condition_guidelines(self, condition_key: str) -> RetrievedChunk | None:
        """Direct condition key lookup."""
        results = self.query(condition_key, n_results=1)
        return results[0] if results else None
