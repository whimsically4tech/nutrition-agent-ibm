"""
tests/test_rag.py
─────────────────────────────────────────────────────────────
Unit tests for the RAG pipeline (ingest + retriever).
Uses an in-memory ChromaDB instance to avoid file I/O.
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def temp_chroma_dir(tmp_path):
    """Provide a fresh temporary ChromaDB directory."""
    return str(tmp_path / "chroma")


@pytest.fixture
def mock_embedding_client():
    """Mock GraniteEmbeddingClient returning fixed-dimension vectors."""
    client = MagicMock()
    # Return a 384-dim zero vector for any input
    client.embed_documents.side_effect = lambda texts: [[0.0] * 384 for _ in texts]
    client.embed_query.return_value = [0.0] * 384
    return client


SAMPLE_FOODS = [
    {
        "food_id": "TEST-001",
        "food_name": "Test Apple",
        "category": "Fruits",
        "serving_size_g": 100,
        "macros": {
            "calories": 52, "protein_g": 0.26, "carbohydrates_g": 13.8,
            "fat_g": 0.17, "fiber_g": 2.4, "sugar_g": 10.4,
            "saturated_fat_g": 0.03, "trans_fat_g": 0.0
        },
        "micros": {
            "vitamin_a_mcg": 3, "vitamin_c_mg": 4.6, "vitamin_d_mcg": 0,
            "vitamin_e_mg": 0.18, "vitamin_k_mcg": 2.2, "vitamin_b12_mcg": 0,
            "folate_mcg": 3, "calcium_mg": 6, "iron_mg": 0.12,
            "magnesium_mg": 5, "potassium_mg": 107, "sodium_mg": 1,
            "zinc_mg": 0.04, "selenium_mcg": 0, "phosphorus_mg": 11
        },
        "glycemic_index": 36,
        "glycemic_load": 5,
        "allergens": []
    },
    {
        "food_id": "TEST-002",
        "food_name": "Test Almond",
        "category": "Nuts",
        "serving_size_g": 100,
        "macros": {
            "calories": 579, "protein_g": 21.2, "carbohydrates_g": 21.6,
            "fat_g": 49.9, "fiber_g": 12.5, "sugar_g": 4.4,
            "saturated_fat_g": 3.8, "trans_fat_g": 0.0
        },
        "micros": {
            "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "vitamin_d_mcg": 0,
            "vitamin_e_mg": 25.6, "vitamin_k_mcg": 0, "vitamin_b12_mcg": 0,
            "folate_mcg": 44, "calcium_mg": 269, "iron_mg": 3.7,
            "magnesium_mg": 270, "potassium_mg": 733, "sodium_mg": 1,
            "zinc_mg": 3.1, "selenium_mcg": 4.1, "phosphorus_mg": 481
        },
        "glycemic_index": 15,
        "glycemic_load": 1,
        "allergens": ["tree_nuts"]
    }
]

SAMPLE_GUIDELINES = {
    "test_condition": {
        "condition": "Test Condition",
        "overview": "A test dietary condition for unit tests.",
        "recommended_foods": ["vegetables", "whole grains"],
        "foods_to_limit": ["sugar", "processed foods"],
        "drug_interactions": [
            {"drug": "TestDrug", "nutrient": "TestNutrient",
             "interaction": "Test interaction description."}
        ]
    }
}


# ─────────────────────────────────────────────────────────────
#  Test: ingest helpers (_build_food_documents, _build_guideline_documents)
# ─────────────────────────────────────────────────────────────

class TestIngestDocumentBuilders:
    def test_build_food_documents_count(self):
        from src.rag.ingest import _build_food_documents
        docs, metas, ids = _build_food_documents(SAMPLE_FOODS)
        assert len(docs) == 2
        assert len(metas) == 2
        assert len(ids) == 2

    def test_build_food_documents_contains_food_name(self):
        from src.rag.ingest import _build_food_documents
        docs, metas, ids = _build_food_documents(SAMPLE_FOODS)
        assert "Test Apple" in docs[0]

    def test_build_food_documents_contains_calories(self):
        from src.rag.ingest import _build_food_documents
        docs, metas, ids = _build_food_documents(SAMPLE_FOODS)
        assert "52" in docs[0]  # Apple calories

    def test_build_food_documents_metadata_food_id(self):
        from src.rag.ingest import _build_food_documents
        docs, metas, ids = _build_food_documents(SAMPLE_FOODS)
        assert metas[0]["food_id"] == "TEST-001"
        assert metas[0]["calories"] == 52

    def test_build_food_documents_allergens_serialised(self):
        from src.rag.ingest import _build_food_documents
        docs, metas, ids = _build_food_documents(SAMPLE_FOODS)
        assert json.loads(metas[1]["allergens"]) == ["tree_nuts"]

    def test_build_guideline_documents_count(self):
        from src.rag.ingest import _build_guideline_documents
        docs, metas, ids = _build_guideline_documents(SAMPLE_GUIDELINES)
        assert len(docs) == 1

    def test_build_guideline_documents_contains_condition(self):
        from src.rag.ingest import _build_guideline_documents
        docs, metas, ids = _build_guideline_documents(SAMPLE_GUIDELINES)
        assert "Test Condition" in docs[0]

    def test_build_guideline_documents_contains_drug_interaction(self):
        from src.rag.ingest import _build_guideline_documents
        docs, metas, ids = _build_guideline_documents(SAMPLE_GUIDELINES)
        assert "TestDrug" in docs[0]


# ─────────────────────────────────────────────────────────────
#  Test: ingest functions (with mocked embeddings + temp files)
# ─────────────────────────────────────────────────────────────

class TestIngestFunctions:
    @patch("src.rag.ingest.get_embedding_client")
    def test_ingest_foods_returns_count(self, mock_get_emb, temp_chroma_dir, tmp_path, mock_embedding_client):
        from src.rag.ingest import ingest_foods, _get_chroma_client

        mock_get_emb.return_value = mock_embedding_client

        # Write temp foods file
        foods_file = tmp_path / "sample_usda_foods.json"
        foods_file.write_text(json.dumps(SAMPLE_FOODS))

        chroma = _get_chroma_client(temp_chroma_dir)

        with patch("src.rag.ingest.FOODS_FILE", foods_file):
            count = ingest_foods(chroma, "test_foods")

        assert count == len(SAMPLE_FOODS)

    @patch("src.rag.ingest.get_embedding_client")
    def test_ingest_guidelines_returns_count(self, mock_get_emb, temp_chroma_dir, tmp_path, mock_embedding_client):
        from src.rag.ingest import ingest_guidelines, _get_chroma_client

        mock_get_emb.return_value = mock_embedding_client

        guidelines_file = tmp_path / "disease_guidelines.json"
        guidelines_file.write_text(json.dumps(SAMPLE_GUIDELINES))

        chroma = _get_chroma_client(temp_chroma_dir)

        with patch("src.rag.ingest.GUIDELINES_FILE", guidelines_file):
            count = ingest_guidelines(chroma, "test_guidelines")

        assert count == len(SAMPLE_GUIDELINES)


# ─────────────────────────────────────────────────────────────
#  Test: FoodRetriever (mocked ChromaDB)
# ─────────────────────────────────────────────────────────────

class TestFoodRetriever:
    @patch("src.rag.retriever.get_embedding_client")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_query_returns_retrieved_chunks(self, mock_chroma_cls, mock_get_emb, mock_embedding_client):
        from src.rag.retriever import FoodRetriever, RetrievedChunk

        mock_get_emb.return_value = mock_embedding_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Food: Test Apple. Calories: 52."]],
            "metadatas": [[{"food_id": "TEST-001", "food_name": "Test Apple", "calories": 52,
                            "protein_g": 0.26, "carbs_g": 13.8, "fat_g": 0.17,
                            "fiber_g": 2.4, "glycemic_index": 36, "allergens": "[]"}]],
            "distances": [[0.1]],
        }
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        mock_chroma_cls.return_value = mock_chroma

        with patch("src.rag.retriever.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = "/tmp/test_chroma"
            mock_settings.return_value.chroma_collection_foods = "test_foods"
            retriever = FoodRetriever()

        results = retriever.query("apple", n_results=1)
        assert len(results) == 1
        assert isinstance(results[0], RetrievedChunk)
        assert results[0].id == "TEST-001"
        assert abs(results[0].score - 0.9) < 0.01  # 1.0 - 0.1 distance

    @patch("src.rag.retriever.get_embedding_client")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_filter_by_allergen_free_excludes_allergens(self, mock_chroma_cls, mock_get_emb, mock_embedding_client):
        from src.rag.retriever import FoodRetriever, RetrievedChunk

        mock_get_emb.return_value = mock_embedding_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Apple.", "Almond."]],
            "metadatas": [[
                {"food_id": "T-001", "food_name": "Apple", "calories": 52, "protein_g": 0,
                 "carbs_g": 14, "fat_g": 0, "fiber_g": 2.4, "glycemic_index": 36, "allergens": "[]"},
                {"food_id": "T-002", "food_name": "Almond", "calories": 579, "protein_g": 21,
                 "carbs_g": 22, "fat_g": 50, "fiber_g": 12, "glycemic_index": 15, "allergens": '["tree_nuts"]'},
            ]],
            "distances": [[0.1, 0.15]],
        }
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        mock_chroma_cls.return_value = mock_chroma

        with patch("src.rag.retriever.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = "/tmp/test_chroma"
            mock_settings.return_value.chroma_collection_foods = "test_foods"
            retriever = FoodRetriever()

        results = retriever.filter_by_allergen_free(
            "high energy snack", excluded_allergens=["tree_nuts"], n_results=5
        )
        food_names = [r.metadata["food_name"] for r in results]
        assert "Apple" in food_names
        assert "Almond" not in food_names


# ─────────────────────────────────────────────────────────────
#  Test: GuidelineRetriever (mocked ChromaDB)
# ─────────────────────────────────────────────────────────────

class TestGuidelineRetriever:
    @patch("src.rag.retriever.get_embedding_client")
    @patch("src.rag.retriever.chromadb.PersistentClient")
    def test_query_returns_guideline_chunk(self, mock_chroma_cls, mock_get_emb, mock_embedding_client):
        from src.rag.retriever import GuidelineRetriever, RetrievedChunk

        mock_get_emb.return_value = mock_embedding_client

        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "documents": [["Condition: Test. Overview: ..."]],
            "metadatas": [[{"condition_key": "test_condition", "condition": "Test Condition"}]],
            "distances": [[0.05]],
        }
        mock_chroma = MagicMock()
        mock_chroma.get_or_create_collection.return_value = mock_collection
        mock_chroma_cls.return_value = mock_chroma

        with patch("src.rag.retriever.get_settings") as mock_settings:
            mock_settings.return_value.chroma_persist_dir = "/tmp/test_chroma"
            mock_settings.return_value.chroma_collection_guidelines = "test_guidelines"
            retriever = GuidelineRetriever()

        results = retriever.query("diabetes diet", n_results=1)
        assert len(results) == 1
        assert results[0].metadata["condition_key"] == "test_condition"
        assert results[0].score > 0.9
