"""
src/agents/knowledge_agent.py
─────────────────────────────────────────────────────────────
Nutrition Knowledge Agent.
Implements RAG over USDA FoodData Central records indexed in ChromaDB.
Retrieves precise calorie, macro, and micronutrient data for foods.
"""
from __future__ import annotations

import json
import logging

from src.models.nutrition import Macronutrients, Micronutrients, NutritionData
from src.rag.retriever import FoodRetriever, RetrievedChunk
from src.services.watsonx_service import get_llm_client

logger = logging.getLogger(__name__)

NUTRITION_EXTRACTION_PROMPT = """\
<|system|>
You are a precise nutrition data extraction assistant. Using ONLY the provided food context, extract structured nutritional data. If a value is not available in the context, use 0.0.

Context:
{context}

<|user|>
Food item: "{food_query}" (portion: {portion_g}g)

Extract the nutritional data per the given portion weight and return a JSON object with this exact structure:
{{
  "food_name": "string",
  "serving_size_g": number,
  "macros": {{
    "calories": number,
    "protein_g": number,
    "carbohydrates_g": number,
    "fat_g": number,
    "fiber_g": number,
    "sugar_g": number,
    "saturated_fat_g": number,
    "trans_fat_g": number
  }},
  "micros": {{
    "vitamin_a_mcg": number, "vitamin_c_mg": number, "vitamin_d_mcg": number,
    "vitamin_e_mg": number, "vitamin_k_mcg": number, "vitamin_b12_mcg": number,
    "folate_mcg": number, "calcium_mg": number, "iron_mg": number,
    "magnesium_mg": number, "potassium_mg": number, "sodium_mg": number,
    "zinc_mg": number, "selenium_mcg": number, "phosphorus_mg": number
  }},
  "glycemic_index": number_or_null,
  "allergens": ["list", "of", "allergens"]
}}

Return ONLY the JSON object, no additional text.
<|assistant|>
"""


class KnowledgeAgent:
    """
    Retrieval-Augmented nutrition lookup agent.
    For a given food item (name + portion size), returns a NutritionData object.
    """

    def __init__(self) -> None:
        self._retriever = FoodRetriever()
        self._llm = get_llm_client()

    def get_nutrition(self, food_query: str, portion_g: float = 100.0) -> NutritionData:
        """
        Look up nutritional data for a food item.

        Args:
            food_query: Natural language food name (e.g., "brown rice cooked")
            portion_g: Portion weight in grams

        Returns:
            NutritionData object for the given portion.
        """
        # 1. Retrieve relevant food context
        chunks = self._retriever.query(food_query, n_results=3)

        if not chunks:
            logger.warning("No RAG results for '%s'; using LLM-only estimation.", food_query)
            return self._estimate_via_llm(food_query, portion_g)

        context = "\n\n".join(c.text for c in chunks)

        # 2. Extract structured nutrition via Granite
        prompt = NUTRITION_EXTRACTION_PROMPT.format(
            context=context,
            food_query=food_query,
            portion_g=portion_g,
        )
        raw = self._llm.generate(prompt)

        # 3. Parse JSON response
        return self._parse_nutrition_response(raw, food_query, portion_g, chunks)

    def get_nutrition_batch(
        self, items: list[tuple[str, float]]
    ) -> list[NutritionData]:
        """
        Look up nutrition for multiple items. Returns list in same order.
        Args:
            items: List of (food_name, portion_g) tuples.
        """
        return [self.get_nutrition(name, grams) for name, grams in items]

    # ── Private helpers ───────────────────────────────────────

    def _parse_nutrition_response(
        self,
        raw: str,
        food_query: str,
        portion_g: float,
        chunks: list[RetrievedChunk],
    ) -> NutritionData:
        """Parse LLM JSON output → NutritionData, with fallback to raw chunk data."""
        # Strip markdown fences
        clean = raw.strip()
        if clean.startswith("```"):
            clean = clean.split("```")[1]
            if clean.startswith("json"):
                clean = clean[4:]
        clean = clean.strip()

        try:
            data = json.loads(clean)
            return NutritionData(
                food_name=data.get("food_name", food_query),
                serving_size_g=data.get("serving_size_g", portion_g),
                macros=Macronutrients(**data.get("macros", {})),
                micros=Micronutrients(**data.get("micros", {})),
                glycemic_index=data.get("glycemic_index"),
                allergens=data.get("allergens", []),
            )
        except (json.JSONDecodeError, TypeError, ValueError) as exc:
            logger.warning("JSON parse failed for '%s': %s. Using metadata.", food_query, exc)
            return self._nutrition_from_metadata(chunks[0], food_query, portion_g)

    def _nutrition_from_metadata(
        self, chunk: RetrievedChunk, food_name: str, portion_g: float
    ) -> NutritionData:
        """Build NutritionData directly from ChromaDB metadata (100g baseline, scaled)."""
        m = chunk.metadata
        factor = portion_g / 100.0
        return NutritionData(
            food_name=m.get("food_name", food_name),
            serving_size_g=portion_g,
            macros=Macronutrients(
                calories=m.get("calories", 0) * factor,
                protein_g=m.get("protein_g", 0) * factor,
                carbohydrates_g=m.get("carbs_g", 0) * factor,
                fat_g=m.get("fat_g", 0) * factor,
                fiber_g=m.get("fiber_g", 0) * factor,
            ),
            glycemic_index=m.get("glycemic_index"),
            allergens=json.loads(m.get("allergens", "[]")) if isinstance(m.get("allergens"), str) else [],
        )

    def _estimate_via_llm(self, food_query: str, portion_g: float) -> NutritionData:
        """Pure LLM estimation when no RAG context is available."""
        prompt = (
            f"<|system|>\nYou are a nutrition database. Estimate nutritional data for:\n"
            f"Food: {food_query} ({portion_g}g)\n"
            "Return ONLY a JSON object with keys: food_name, serving_size_g, macros "
            "(calories, protein_g, carbohydrates_g, fat_g, fiber_g, sugar_g, saturated_fat_g, trans_fat_g), "
            "micros (vitamin_a_mcg, vitamin_c_mg, vitamin_d_mcg, vitamin_e_mg, vitamin_k_mcg, "
            "vitamin_b12_mcg, folate_mcg, calcium_mg, iron_mg, magnesium_mg, potassium_mg, "
            "sodium_mg, zinc_mg, selenium_mcg, phosphorus_mg), glycemic_index, allergens.\n"
            "<|assistant|>"
        )
        raw = self._llm.generate(prompt)
        return self._parse_nutrition_response(raw, food_query, portion_g, [])
