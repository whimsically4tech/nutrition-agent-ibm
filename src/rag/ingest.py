"""
src/rag/ingest.py
─────────────────────────────────────────────────────────────
Vector DB indexing script.
Reads seed JSON files, generates Granite embeddings, and indexes
them into ChromaDB collections.

Run once (or when data changes):
    python -m src.rag.ingest
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import chromadb
from chromadb.config import Settings as ChromaSettings

from src.config import get_settings
from src.services.watsonx_service import get_embedding_client

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parents[2] / "data"
FOODS_FILE = DATA_DIR / "sample_usda_foods.json"
GUIDELINES_FILE = DATA_DIR / "disease_guidelines.json"


def _get_chroma_client(persist_dir: str) -> chromadb.ClientAPI:
    Path(persist_dir).mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(
        path=persist_dir,
        settings=ChromaSettings(anonymized_telemetry=False),
    )


def _build_food_documents(foods: list[dict]) -> tuple[list[str], list[dict], list[str]]:
    """Return (documents, metadatas, ids) for ChromaDB from USDA food records."""
    documents, metadatas, ids = [], [], []
    for food in foods:
        macros = food.get("macros", {})
        micros = food.get("micros", {})
        text = (
            f"Food: {food['food_name']}. "
            f"Category: {food.get('category', 'Unknown')}. "
            f"Per 100g — Calories: {macros.get('calories', 0)} kcal, "
            f"Protein: {macros.get('protein_g', 0)}g, "
            f"Carbs: {macros.get('carbohydrates_g', 0)}g, "
            f"Fat: {macros.get('fat_g', 0)}g, "
            f"Fiber: {macros.get('fiber_g', 0)}g. "
            f"Glycemic Index: {food.get('glycemic_index', 'N/A')}. "
            f"Allergens: {', '.join(food.get('allergens', [])) or 'None'}. "
            f"Vitamin C: {micros.get('vitamin_c_mg', 0)}mg, "
            f"Iron: {micros.get('iron_mg', 0)}mg, "
            f"Calcium: {micros.get('calcium_mg', 0)}mg, "
            f"Sodium: {micros.get('sodium_mg', 0)}mg, "
            f"Potassium: {micros.get('potassium_mg', 0)}mg."
        )
        documents.append(text)
        metadatas.append({
            "food_id": food.get("food_id", ""),
            "food_name": food["food_name"],
            "category": food.get("category", ""),
            "calories": macros.get("calories", 0),
            "protein_g": macros.get("protein_g", 0),
            "carbs_g": macros.get("carbohydrates_g", 0),
            "fat_g": macros.get("fat_g", 0),
            "fiber_g": macros.get("fiber_g", 0),
            "glycemic_index": food.get("glycemic_index") or 0,
            "allergens": json.dumps(food.get("allergens", [])),
        })
        ids.append(food.get("food_id", food["food_name"]))
    return documents, metadatas, ids


def _build_guideline_documents(
    guidelines: dict,
) -> tuple[list[str], list[dict], list[str]]:
    """Return (documents, metadatas, ids) for ChromaDB from disease guideline records."""
    documents, metadatas, ids = [], [], []
    for condition_key, guideline in guidelines.items():
        # Main condition overview
        text = (
            f"Condition: {guideline['condition']}. "
            f"Overview: {guideline['overview']} "
            f"Recommended foods: {', '.join(guideline.get('recommended_foods', []))}. "
            f"Foods to limit: {', '.join(guideline.get('foods_to_limit', []))}. "
        )
        # Append drug interactions as inline text
        for di in guideline.get("drug_interactions", []):
            text += (
                f"Drug-nutrient interaction: {di['drug']} with {di['nutrient']}: "
                f"{di['interaction']}. "
            )
        documents.append(text)
        metadatas.append({
            "condition_key": condition_key,
            "condition": guideline["condition"],
        })
        ids.append(f"guideline-{condition_key}")

    return documents, metadatas, ids


class EmbeddingFunction(chromadb.EmbeddingFunction):
    """Adapter: wraps GraniteEmbeddingClient for ChromaDB."""

    def __call__(self, input: chromadb.Documents) -> chromadb.Embeddings:  # noqa: A002
        client = get_embedding_client()
        return client.embed_documents(list(input))


def ingest_foods(chroma_client: chromadb.ClientAPI, collection_name: str) -> int:
    """Index USDA food records. Returns count of indexed documents."""
    with FOODS_FILE.open() as f:
        foods: list[dict] = json.load(f)

    documents, metadatas, ids = _build_food_documents(foods)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=EmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )

    # Upsert to handle re-runs gracefully
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info("Indexed %d food records into '%s'", len(documents), collection_name)
    return len(documents)


def ingest_guidelines(chroma_client: chromadb.ClientAPI, collection_name: str) -> int:
    """Index disease guidelines. Returns count of indexed documents."""
    with GUIDELINES_FILE.open() as f:
        guidelines: dict = json.load(f)

    documents, metadatas, ids = _build_guideline_documents(guidelines)
    collection = chroma_client.get_or_create_collection(
        name=collection_name,
        embedding_function=EmbeddingFunction(),
        metadata={"hnsw:space": "cosine"},
    )
    collection.upsert(documents=documents, metadatas=metadatas, ids=ids)
    logger.info(
        "Indexed %d guideline documents into '%s'", len(documents), collection_name
    )
    return len(documents)


def run_ingest() -> None:
    """Main entry point: index all data sources."""
    cfg = get_settings()
    chroma = _get_chroma_client(cfg.chroma_persist_dir)

    total_foods = ingest_foods(chroma, cfg.chroma_collection_foods)
    total_guidelines = ingest_guidelines(chroma, cfg.chroma_collection_guidelines)

    print(f"✅ Ingestion complete: {total_foods} foods, {total_guidelines} guidelines")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    run_ingest()
