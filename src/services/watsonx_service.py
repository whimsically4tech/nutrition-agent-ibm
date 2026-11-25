"""
src/services/watsonx_service.py
─────────────────────────────────────────────────────────────
Wraps the IBM watsonx.ai SDK to provide:
  - Text generation via IBM Granite 3 (ibm/granite-3-8b-instruct)
  - Multimodal image analysis via Granite Vision
  - Embedding generation via Granite Embedding
  - Safety evaluation via Granite Guardian

All clients are built as cached singletons to avoid re-auth overhead.
"""
from __future__ import annotations

import base64
import logging
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config import Settings, get_settings

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Low-level credential factory
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _get_api_client(settings: Settings | None = None) -> Any:
    """Return a cached watsonx APIClient."""
    from ibm_watsonx_ai import APIClient, Credentials  # lazy import
    cfg = settings or get_settings()
    credentials = Credentials(
        url=cfg.watsonx_url,
        api_key=cfg.watsonx_apikey,
    )
    return APIClient(credentials=credentials, project_id=cfg.watsonx_project_id)


# ─────────────────────────────────────────────────────────────
#  LLM client (Granite 3 text generation)
# ─────────────────────────────────────────────────────────────

class GraniteLLMClient:
    """Thin wrapper around ModelInference for text generation."""

    def __init__(self, settings: Settings | None = None) -> None:
        from ibm_watsonx_ai.foundation_models import ModelInference  # lazy
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # lazy
        self._cfg = settings or get_settings()
        self._model = ModelInference(
            model_id=self._cfg.granite_llm_model,
            api_client=_get_api_client(self._cfg),
            params={
                GenParams.DECODING_METHOD: self._cfg.llm_params["decoding_method"],
                GenParams.MAX_NEW_TOKENS: self._cfg.llm_params["max_new_tokens"],
                GenParams.MIN_NEW_TOKENS: self._cfg.llm_params["min_new_tokens"],
                GenParams.TEMPERATURE: self._cfg.llm_params["temperature"],
                GenParams.REPETITION_PENALTY: self._cfg.llm_params["repetition_penalty"],
            },
        )

    def generate(self, prompt: str, **override_params: Any) -> str:
        """Generate text from a prompt. Returns the generated string."""
        try:
            if override_params:
                resp = self._model.generate_text(prompt, params=override_params)
            else:
                resp = self._model.generate_text(prompt)
            return resp.strip()
        except Exception as exc:
            logger.error("LLM generation failed: %s", exc)
            raise

    def generate_with_messages(self, messages: list[dict]) -> str:
        """Chat-style generation using a list of role/content dicts."""
        # Flatten to a simple prompt using Granite's instruction format
        prompt_parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            if role == "system":
                prompt_parts.append(f"<|system|>\n{content}")
            elif role == "user":
                prompt_parts.append(f"<|user|>\n{content}")
            elif role == "assistant":
                prompt_parts.append(f"<|assistant|>\n{content}")
        prompt_parts.append("<|assistant|>")
        return self.generate("\n".join(prompt_parts))


# ─────────────────────────────────────────────────────────────
#  Vision client (Granite Vision multimodal)
# ─────────────────────────────────────────────────────────────

class GraniteVisionClient:
    """
    Wrapper for Granite Vision multimodal inference.
    Accepts an image file path or raw bytes and a text prompt.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        from ibm_watsonx_ai.foundation_models import ModelInference  # lazy
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # lazy
        self._cfg = settings or get_settings()
        self._model = ModelInference(
            model_id=self._cfg.granite_vision_model,
            api_client=_get_api_client(self._cfg),
            params={
                GenParams.MAX_NEW_TOKENS: 1024,
                GenParams.TEMPERATURE: 0.3,
            },
        )

    def analyze_image(
        self,
        image_source: str | Path | bytes,
        prompt: str = "Identify all food items and estimate portion sizes in this meal image.",
    ) -> str:
        """
        Send an image and prompt to Granite Vision.
        Returns the model's textual description.
        """
        # Encode image to base64 data URI
        if isinstance(image_source, (str, Path)):
            image_path = Path(image_source)
            with image_path.open("rb") as f:
                raw_bytes = f.read()
            suffix = image_path.suffix.lstrip(".").lower()
            mime = "image/jpeg" if suffix in ("jpg", "jpeg") else f"image/{suffix}"
        else:
            raw_bytes = image_source
            mime = "image/jpeg"

        b64 = base64.b64encode(raw_bytes).decode()
        data_uri = f"data:{mime};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": data_uri}},
                    {"type": "text", "text": prompt},
                ],
            }
        ]

        try:
            resp = self._model.chat(messages=messages)
            # Extract text from response
            if isinstance(resp, dict):
                return (
                    resp.get("results", [{}])[0]
                    .get("generated_text", "")
                    .strip()
                )
            return str(resp).strip()
        except Exception as exc:
            logger.error("Vision inference failed: %s", exc)
            raise


# ─────────────────────────────────────────────────────────────
#  Embedding client (Granite Embedding)
# ─────────────────────────────────────────────────────────────

class GraniteEmbeddingClient:
    """Generate dense embeddings for RAG indexing and retrieval."""

    def __init__(self, settings: Settings | None = None) -> None:
        from ibm_watsonx_ai.foundation_models.embeddings import Embeddings  # lazy
        self._cfg = settings or get_settings()
        self._embedder = Embeddings(
            model_id=self._cfg.granite_embedding_model,
            api_client=_get_api_client(self._cfg),
            params=self._cfg.embedding_params,
        )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Return a list of embedding vectors for a list of texts."""
        try:
            resp = self._embedder.embed_documents(texts=texts)
            return resp
        except Exception as exc:
            logger.error("Embedding failed: %s", exc)
            raise

    def embed_query(self, text: str) -> list[float]:
        """Return a single embedding vector for a query string."""
        return self.embed_documents([text])[0]


# ─────────────────────────────────────────────────────────────
#  Guardian client (Granite Guardian safety)
# ─────────────────────────────────────────────────────────────

class GraniteGuardianClient:
    """
    Uses Granite Guardian to detect harmful / unsafe content.
    Returns a safety verdict and risk category.
    """

    GUARDIAN_PROMPT_TEMPLATE = (
        "<|start_of_role|>user<|end_of_role|>"
        "You are a safety classifier. Assess whether the following text contains "
        "harmful medical claims, misinformation, or advice that could endanger health.\n\n"
        "Text to evaluate:\n{text}\n\n"
        "Respond ONLY with a JSON object: "
        '{{\"safe\": true/false, \"risk_category\": \"string\", \"reason\": \"string\"}}'
        "<|end_of_text|><|start_of_role|>assistant<|end_of_role|>"
    )

    def __init__(self, settings: Settings | None = None) -> None:
        from ibm_watsonx_ai.foundation_models import ModelInference  # lazy
        from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams  # lazy
        self._cfg = settings or get_settings()
        self._model = ModelInference(
            model_id=self._cfg.granite_guardian_model,
            api_client=_get_api_client(self._cfg),
            params={
                GenParams.MAX_NEW_TOKENS: 256,
                GenParams.DECODING_METHOD: "greedy",
            },
        )

    def check(self, text: str) -> dict[str, Any]:
        """
        Evaluate text for safety.

        Returns:
            dict with keys: safe (bool), risk_category (str), reason (str)
        """
        import json

        prompt = self.GUARDIAN_PROMPT_TEMPLATE.format(text=text)
        try:
            raw = self._model.generate_text(prompt).strip()
            # Parse JSON response
            # Strip markdown fences if present
            if raw.startswith("```"):
                raw = raw.split("```")[1]
                if raw.startswith("json"):
                    raw = raw[4:]
            return json.loads(raw)
        except Exception as exc:
            logger.warning("Guardian check failed or non-JSON response: %s", exc)
            # Fail open with a warning — do not block legitimate advice
            return {"safe": True, "risk_category": "unknown", "reason": str(exc)}


# ─────────────────────────────────────────────────────────────
#  Demo / Mock mode — used when IBM credentials are placeholders
# ─────────────────────────────────────────────────────────────

_PLACEHOLDER_KEYS = {"your_ibm_cloud_api_key_here", "", "placeholder", "test-api-key"}


def _is_demo_mode() -> bool:
    """Return True when running without real IBM credentials."""
    cfg = get_settings()
    return cfg.watsonx_apikey.strip() in _PLACEHOLDER_KEYS


class MockLLMClient:
    """
    Returns realistic pre-canned responses for every generate() call
    so the entire UI works end-to-end without a real IBM account.
    """

    # ── Canned response library ───────────────────────────────
    _NUTRITION_JSON = """\
{
  "food_name": "{food}",
  "serving_size_g": {grams},
  "macros": {{"calories": {cal}, "protein_g": {prot}, "carbohydrates_g": {carb},
    "fat_g": {fat}, "fiber_g": {fiber}, "sugar_g": {sugar},
    "saturated_fat_g": 1.2, "trans_fat_g": 0.0}},
  "micros": {{"vitamin_a_mcg": 45, "vitamin_c_mg": 12.0, "vitamin_d_mcg": 1.5,
    "vitamin_e_mg": 0.9, "vitamin_k_mcg": 8.0, "vitamin_b12_mcg": 0.4,
    "folate_mcg": 22, "calcium_mg": 48, "iron_mg": 1.2, "magnesium_mg": 18,
    "potassium_mg": 210, "sodium_mg": 95, "zinc_mg": 0.6,
    "selenium_mcg": 8.0, "phosphorus_mg": 95}},
  "glycemic_index": 45,
  "allergens": []
}}"""

    _MEAL_PLAN_JSON = """\
{
  "plan_name": "7-Day Balanced Wellness Plan (Demo)",
  "daily_calorie_target": 2000,
  "weekly_summary": "A demo balanced plan rich in whole foods, lean proteins, and fibre.",
  "days": [
    {"day":1,"day_name":"Monday","total_calories":1980,"meals":[
      {"meal_type":"breakfast","meal_name":"Oats & Berries","items":[
        {"food_name":"Rolled oats","quantity_g":80,"description":"cooked in water"},
        {"food_name":"Mixed berries","quantity_g":100,"description":"fresh"}],
       "estimated_calories":310,"notes":"Add a teaspoon of honey for sweetness."},
      {"meal_type":"lunch","meal_name":"Grilled Chicken Salad","items":[
        {"food_name":"Chicken breast","quantity_g":150,"description":"grilled"},
        {"food_name":"Mixed greens","quantity_g":100,"description":"fresh"},
        {"food_name":"Olive oil","quantity_g":10,"description":"dressing"}],
       "estimated_calories":320,"notes":"Squeeze lemon for extra flavour."},
      {"meal_type":"dinner","meal_name":"Salmon & Brown Rice","items":[
        {"food_name":"Salmon fillet","quantity_g":180,"description":"baked"},
        {"food_name":"Brown rice","quantity_g":180,"description":"cooked"},
        {"food_name":"Broccoli","quantity_g":120,"description":"steamed"}],
       "estimated_calories":590,"notes":"Season with herbs instead of salt."},
      {"meal_type":"snack","meal_name":"Apple & Almonds","items":[
        {"food_name":"Apple","quantity_g":150,"description":"fresh"},
        {"food_name":"Almonds","quantity_g":30,"description":"unsalted"}],
       "estimated_calories":200,"notes":"Great pre-workout snack."}]},
    {"day":2,"day_name":"Tuesday","total_calories":1960,"meals":[
      {"meal_type":"breakfast","meal_name":"Scrambled Eggs & Toast","items":[
        {"food_name":"Whole eggs","quantity_g":180,"description":"scrambled"},
        {"food_name":"Whole wheat bread","quantity_g":60,"description":"toasted"}],
       "estimated_calories":340,"notes":"Use olive oil spray instead of butter."},
      {"meal_type":"lunch","meal_name":"Lentil Soup","items":[
        {"food_name":"Red lentils","quantity_g":100,"description":"cooked"},
        {"food_name":"Carrots","quantity_g":80,"description":"diced"},
        {"food_name":"Spinach","quantity_g":50,"description":"fresh"}],
       "estimated_calories":280,"notes":"High in plant protein and iron."},
      {"meal_type":"dinner","meal_name":"Turkey & Vegetables","items":[
        {"food_name":"Turkey breast","quantity_g":150,"description":"roasted"},
        {"food_name":"Sweet potato","quantity_g":150,"description":"baked"},
        {"food_name":"Green beans","quantity_g":100,"description":"steamed"}],
       "estimated_calories":420,"notes":"Rich in vitamins A and C."},
      {"meal_type":"snack","meal_name":"Greek Yogurt","items":[
        {"food_name":"Greek yogurt","quantity_g":200,"description":"plain low-fat"}],
       "estimated_calories":130,"notes":"Add cinnamon for extra flavour."}]}
  ],
  "grocery_list": {
    "proteins": ["Chicken breast","Salmon fillet","Turkey breast","Whole eggs","Greek yogurt","Red lentils"],
    "grains": ["Rolled oats","Brown rice","Whole wheat bread","Sweet potato"],
    "vegetables": ["Mixed greens","Broccoli","Carrots","Spinach","Green beans"],
    "fruits": ["Mixed berries","Apple"],
    "fats": ["Almonds","Olive oil"]
  },
  "substitution_options": [
    "Swap salmon for canned tuna (more affordable, similar omega-3 content).",
    "Replace brown rice with quinoa for higher protein content.",
    "Use tofu instead of chicken for a vegan protein source.",
    "Swap cow\\'s milk yogurt for coconut yogurt (dairy-free option)."
  ]
}"""

    # Note: single braces are literal in this raw string — no .format() used here
    # We build the JSON programmatically instead
    _FOOD_ITEMS_JSON = None  # unused; food items built dynamically in generate()

    _HEALTH_ADVICE = """\
## Dietary Recommendations (Demo Mode)

Based on your profile, here are evidence-based dietary recommendations:

### Foods to Prioritise
- **Non-starchy vegetables** (broccoli, spinach, peppers) — high in fibre and micronutrients
- **Lean proteins** (chicken, fish, legumes) — support muscle and satiety
- **Whole grains** (oats, brown rice, quinoa) — sustained energy, lower GI

### Foods to Limit
- Refined sugars and ultra-processed foods
- High-sodium packaged snacks (>600mg sodium per serving)
- Saturated fats from red meat and full-fat dairy

### Practical Tips
1. Eat at regular intervals (every 3-4 hours) to maintain blood sugar stability
2. Aim for half your plate to be vegetables at every main meal
3. Stay hydrated — 2-3 litres of water daily

> ⚕️ **Medical Disclaimer:** This is demo content generated without a live IBM Granite connection. \
Consult a qualified healthcare professional for personalised medical or dietary advice."""

    _FEEDBACK = """\
1. Good meal choice — includes proteins and carbohydrates for balanced energy.
2. Watch portion sizes to stay within your daily calorie target.
3. Consider adding a vegetable side at your next meal to boost fibre intake."""

    # ── Interface ─────────────────────────────────────────────

    def generate(self, prompt: str, **_: Any) -> str:  # noqa: ARG002
        import json
        import re

        p = prompt.lower()

        # ── 1. Food item extraction (FOOD_PARSE_PROMPT) ───────
        # Triggered by presence of "food_name" and "quantity_g" in the prompt template
        if "food_name" in p and "quantity_g" in p and "meal description" in p:
            # Extract the meal description text from the prompt
            desc_match = re.search(
                r'[Mm]eal description[:\s]*["\']?([^"\'\n<|]{3,200})', prompt
            )
            description = desc_match.group(1).strip().strip('"\'') if desc_match else "food"
            # Split on commas/and to get individual items
            raw_items = [
                x.strip().strip('"\'') for x in re.split(r'[,&]|\band\b', description) if x.strip()
            ]
            if not raw_items:
                raw_items = [description]
            # Assign realistic portion weights per item type
            _PORTION_DEFAULTS = {
                "bread": 60, "toast": 60, "banana": 120, "smoothie": 300,
                "dry fruit": 30, "dryfruits": 30, "dryfruit": 30, "nuts": 30,
                "rice": 180, "egg": 120, "milk": 240, "juice": 240,
                "oat": 80, "chicken": 150, "fish": 150, "salmon": 150,
                "apple": 150, "orange": 180, "yogurt": 200, "cheese": 50,
                "butter": 15, "oil": 10, "coffee": 240, "tea": 240,
            }
            items = []
            for item in raw_items[:6]:  # cap at 6 items
                name = item.strip()
                if not name:
                    continue
                qty = 100
                for key, default_g in _PORTION_DEFAULTS.items():
                    if key in name.lower():
                        qty = default_g
                        break
                items.append({
                    "food_name": name,
                    "quantity_g": qty,
                    "confidence": 0.85,
                })
            return json.dumps(items)

        # ── 2. Nutrition data extraction ──────────────────────
        if "serving_size_g" in p or ("macros" in p and "micros" in p):
            food_match = re.search(r'[Ff]ood(?:\s+item)?[:\s]+"?([^"\n,<|]+)', prompt)
            food = food_match.group(1).strip() if food_match else "food item"
            grams_match = re.search(r'portion[:\s]+(\d+(?:\.\d+)?)\s*g', prompt)
            if not grams_match:
                grams_match = re.search(r'(\d+(?:\.\d+)?)\s*g', prompt)
            g = float(grams_match.group(1)) if grams_match else 100.0
            factor = g / 100.0
            result = {
                "food_name": food,
                "serving_size_g": g,
                "macros": {
                    "calories": round(150 * factor, 1),
                    "protein_g": round(8 * factor, 1),
                    "carbohydrates_g": round(20 * factor, 1),
                    "fat_g": round(5 * factor, 1),
                    "fiber_g": round(3 * factor, 1),
                    "sugar_g": round(6 * factor, 1),
                    "saturated_fat_g": round(1.2 * factor, 1),
                    "trans_fat_g": 0.0,
                },
                "micros": {
                    "vitamin_a_mcg": round(45 * factor, 1),
                    "vitamin_c_mg": round(12.0 * factor, 1),
                    "vitamin_d_mcg": round(1.5 * factor, 1),
                    "vitamin_e_mg": round(0.9 * factor, 1),
                    "vitamin_k_mcg": round(8.0 * factor, 1),
                    "vitamin_b12_mcg": round(0.4 * factor, 2),
                    "folate_mcg": round(22 * factor, 1),
                    "calcium_mg": round(48 * factor, 1),
                    "iron_mg": round(1.2 * factor, 2),
                    "magnesium_mg": round(18 * factor, 1),
                    "potassium_mg": round(210 * factor, 1),
                    "sodium_mg": round(95 * factor, 1),
                    "zinc_mg": round(0.6 * factor, 2),
                    "selenium_mcg": round(8.0 * factor, 1),
                    "phosphorus_mg": round(95 * factor, 1),
                },
                "glycemic_index": 45,
                "allergens": [],
            }
            return json.dumps(result)

        # ── 3. Meal plan ──────────────────────────────────────
        if "meal plan" in p or "7-day" in p or "weekly" in p or ("generate" in p and "meal" in p):
            return self._MEAL_PLAN_JSON

        # ── 4. Feedback ───────────────────────────────────────
        if "feedback" in p or "remaining" in p or "consumed" in p:
            return self._FEEDBACK

        # ── 5. Health advice ──────────────────────────────────
        if any(w in p for w in ("health", "condition", "clinical", "diabetes",
                                "hypertension", "kidney", "cholesterol", "diet")):
            return self._HEALTH_ADVICE

        # ── 6. Intent classification ──────────────────────────
        if any(w in p for w in ("classify", "intent", "category", "categories")):
            if any(w in p for w in ("log", "ate", "eaten", "just had", "just ate")):
                return "log_meal"
            if any(w in p for w in ("plan", "week", "schedule")):
                return "meal_plan"
            if any(w in p for w in ("calor", "protein", "nutrient", "nutrition")):
                return "nutrition_lookup"
            if any(w in p for w in ("diabetes", "hypertension", "condition", "medical")):
                return "health_advice"
            return "general"

        # ── 7. Guardian safety check ──────────────────────────
        if any(w in p for w in ("safety", "harmful", "classifier", "assess", "evaluate")):
            return json.dumps({
                "safe": True,
                "risk_category": "none",
                "reason": "Demo mode — safety check skipped.",
            })

        # ── 8. Substitutions ──────────────────────────────────
        if "substitut" in p:
            return json.dumps([
                "Brown rice instead of white rice",
                "Quinoa for added protein",
                "Cauliflower rice for low-carb option",
                "Barley for more fibre",
                "Oats as a wholesome breakfast alternative",
            ])

        # ── Fallback ──────────────────────────────────────────
        return (
            "**Demo Mode — IBM Granite not connected**\n\n"
            "This is a placeholder response. To activate real AI, "
            "add your IBM watsonx credentials to the `.env` file:\n"
            "```\nWATSONX_APIKEY=your_real_key\nWATSONX_PROJECT_ID=your_project_id\n```"
        )

    def generate_with_messages(self, messages: list[dict]) -> str:
        combined = " ".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        return self.generate(combined)


class MockEmbeddingClient:
    """Returns deterministic pseudo-embeddings (zero vector) in demo mode."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        import hashlib
        result = []
        for text in texts:
            # Deterministic non-zero vector seeded from text hash
            seed = int(hashlib.md5(text.encode()).hexdigest(), 16) % (2**31)
            import random
            rng = random.Random(seed)
            result.append([rng.uniform(-0.1, 0.1) for _ in range(384)])
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


class MockVisionClient:
    """Returns realistic food identification in demo mode."""

    def analyze_image(self, image_source: Any, prompt: str = "") -> str:
        return (
            '[{"food_name": "rice bowl", "quantity_g": 200, "confidence": 0.85}, '
            '{"food_name": "grilled chicken", "quantity_g": 150, "confidence": 0.90}, '
            '{"food_name": "steamed broccoli", "quantity_g": 80, "confidence": 0.80}]'
        )


class MockGuardianClient:
    """Always returns safe in demo mode."""

    def check(self, text: str) -> dict[str, Any]:  # noqa: ARG002
        return {"safe": True, "risk_category": "none", "reason": "Demo mode."}


# ─────────────────────────────────────────────────────────────
#  Convenience singleton accessors
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=1)
def get_llm_client():
    if _is_demo_mode():
        logger.warning("⚠️  Demo mode: returning MockLLMClient (no IBM credentials)")
        return MockLLMClient()
    return GraniteLLMClient()


@lru_cache(maxsize=1)
def get_vision_client():
    if _is_demo_mode():
        return MockVisionClient()
    return GraniteVisionClient()


@lru_cache(maxsize=1)
def get_embedding_client():
    if _is_demo_mode():
        return MockEmbeddingClient()
    return GraniteEmbeddingClient()


@lru_cache(maxsize=1)
def get_guardian_client():
    if _is_demo_mode():
        return MockGuardianClient()
    return GraniteGuardianClient()
