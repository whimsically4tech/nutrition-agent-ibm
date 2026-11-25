"""
src/agents/food_log_agent.py
─────────────────────────────────────────────────────────────
Food Log & Feedback Agent.
Handles multimodal meal ingestion:
  - Text: NLP parsing of natural language descriptions
  - Audio: Speech-to-Text → text parsing pipeline
  - Image: Granite Vision → food identification → nutrition lookup

After ingestion, deducts from daily macro budget and generates
real-time personalised feedback alerts.
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from src.agents.guardrails import GuardrailsFilter
from src.agents.knowledge_agent import KnowledgeAgent
from src.models.meal import DailySummary, MacroBudget, MealItem, MealLog, MealSource, MealType
from src.models.nutrition import Macronutrients, Micronutrients
from src.models.user import UserProfile
from src.services.stt_service import SpeechToTextService
from src.services.watsonx_service import get_llm_client, get_vision_client

logger = logging.getLogger(__name__)

FOOD_PARSE_PROMPT = """\
<|system|>
You are a nutrition assistant that extracts food items from meal descriptions.
<|user|>
Meal description: "{description}"

Extract individual food items and estimate realistic portion weights in grams.
Return ONLY a JSON array:
[
  {{"food_name": "string", "quantity_g": number, "confidence": number_0_to_1}},
  ...
]
Use these guidelines:
- 1 slice bread ≈ 30g, 1 egg ≈ 60g, 1 cup cooked rice ≈ 180g
- 1 glass juice ≈ 240g, 1 tablespoon butter ≈ 14g
- Express confidence as 1.0 for explicit amounts, 0.7 for estimated
<|assistant|>
"""

FEEDBACK_PROMPT = """\
<|system|>
You are a helpful nutrition coach providing brief, actionable meal feedback.
<|user|>
User: {name}, Goal: {goal}, Daily target: {target_cal:.0f} kcal
Consumed today: {consumed_cal:.0f} kcal ({consumed_pct:.0f}% of target)
Protein: {protein_g:.0f}g | Carbs: {carbs_g:.0f}g | Fat: {fat_g:.0f}g | Sodium: {sodium_mg:.0f}mg

Just logged ({meal_type}):
{meal_items}

Remaining for today: {remaining_cal:.0f} kcal

Provide 2-3 brief, specific feedback points:
1. Comment on this meal (positive or constructive)
2. Alert for any nutrient concerns (high sodium > 800mg/meal, low protein, etc.)
3. Suggestion for the next meal to balance the day

Keep each point to 1-2 sentences. Be encouraging.
<|assistant|>
"""

VISION_FOOD_PROMPT = (
    "Identify all food items visible in this meal image. "
    "For each item, estimate the portion size in grams based on visual cues "
    "(plate size, food volume, typical serving sizes). "
    "Return a JSON array: "
    '[{"food_name": "string", "quantity_g": number, "confidence": number_0_to_1}]'
    "\nReturn ONLY the JSON array."
)


class FoodLogAgent:
    """
    Multimodal food logging agent with real-time nutritional feedback.
    """

    def __init__(self) -> None:
        self._knowledge = KnowledgeAgent()
        self._llm = get_llm_client()
        self._vision = get_vision_client()
        self._stt = SpeechToTextService()
        self._guardrails = GuardrailsFilter(use_llm_guardian=False)  # speed optimised

    # ── Public API ────────────────────────────────────────────

    def log_meal_text(
        self,
        profile: UserProfile,
        description: str,
        meal_type: MealType = MealType.SNACK,
        daily_summary: DailySummary | None = None,
    ) -> MealLog:
        """Log a meal from a natural language text description."""
        items = self._parse_food_text(description)
        return self._build_meal_log(
            profile, items, MealSource.TEXT, description, meal_type, daily_summary
        )

    def log_meal_audio(
        self,
        profile: UserProfile,
        audio_bytes: bytes,
        content_type: str = "audio/wav",
        meal_type: MealType = MealType.SNACK,
        daily_summary: DailySummary | None = None,
    ) -> MealLog:
        """Transcribe audio and log the described meal."""
        transcript = self._stt.transcribe(audio_bytes, content_type)
        logger.info("Audio transcript: %s", transcript)
        return self.log_meal_text(profile, transcript, meal_type, daily_summary)

    def log_meal_image(
        self,
        profile: UserProfile,
        image_data: bytes | str | Path,
        meal_type: MealType = MealType.SNACK,
        daily_summary: DailySummary | None = None,
    ) -> MealLog:
        """Analyse a plate photo and log the identified foods."""
        raw_vision = self._vision.analyze_image(image_data, VISION_FOOD_PROMPT)
        items = self._parse_items_json(raw_vision)
        description = f"[Image analysis]: {raw_vision[:200]}"
        return self._build_meal_log(
            profile, items, MealSource.IMAGE, description, meal_type, daily_summary
        )

    def generate_feedback(
        self,
        profile: UserProfile,
        meal_log: MealLog,
        daily_summary: DailySummary,
    ) -> list[str]:
        """Generate real-time feedback alerts for a logged meal."""
        consumed = daily_summary.consumed
        target = daily_summary.target
        consumed_pct = (consumed.calories / target.calories * 100) if target.calories else 0
        remaining_cal = max(0, target.calories - consumed.calories)

        meal_items_str = "\n".join(
            f"  - {item.food_name}: {item.quantity_g}g"
            for item in meal_log.items
        )

        prompt = FEEDBACK_PROMPT.format(
            name=profile.name,
            goal=profile.fitness_goal.value,
            target_cal=target.calories,
            consumed_cal=consumed.calories,
            consumed_pct=consumed_pct,
            protein_g=consumed.protein_g,
            carbs_g=consumed.carbohydrates_g,
            fat_g=consumed.fat_g,
            sodium_mg=0,  # would come from micronutrient totals
            meal_type=meal_log.meal_type.value,
            meal_items=meal_items_str or "No items detected",
            remaining_cal=remaining_cal,
        )

        raw = self._llm.generate(prompt)
        # Split numbered list into individual alerts
        alerts = [
            line.strip().lstrip("123.-) ").strip()
            for line in raw.split("\n")
            if line.strip() and len(line.strip()) > 10
        ]
        return alerts[:5]  # Cap at 5 alerts

    def build_daily_summary(
        self,
        profile: UserProfile,
        meal_logs: list[MealLog],
        date: str | None = None,
    ) -> DailySummary:
        """
        Aggregate all meal logs for a day into a DailySummary.
        """
        from src.agents.recommendation_agent import RecommendationAgent

        rec_agent = RecommendationAgent()
        macro_targets = rec_agent.calculate_macros_summary(profile)
        target = MacroBudget(
            calories=macro_targets["calories"],
            protein_g=macro_targets["protein_g"],
            carbohydrates_g=macro_targets["carbohydrates_g"],
            fat_g=macro_targets["fat_g"],
            fiber_g=macro_targets["fiber_g"],
        )

        # Aggregate consumed macros
        agg: dict[str, float] = {k: 0.0 for k in Macronutrients.model_fields}
        agg_micros: dict[str, float] = {k: 0.0 for k in Micronutrients.model_fields}
        for log in meal_logs:
            for k, v in log.total_macros.model_dump().items():
                agg[k] += v
            for meal_item in log.items:
                if meal_item.nutrition:
                    for k, v in meal_item.nutrition.micros.model_dump().items():
                        agg_micros[k] += v

        consumed = Macronutrients(**agg)
        remaining = MacroBudget(
            calories=max(0, target.calories - consumed.calories),
            protein_g=max(0, target.protein_g - consumed.protein_g),
            carbohydrates_g=max(0, target.carbohydrates_g - consumed.carbohydrates_g),
            fat_g=max(0, target.fat_g - consumed.fat_g),
            fiber_g=max(0, target.fiber_g - consumed.fiber_g),
        )

        # Build feedback
        feedback = self._generate_daily_feedback(profile, consumed, target)

        return DailySummary(
            user_id=profile.user_id,
            date=date or datetime.utcnow().date().isoformat(),
            target=target,
            consumed=consumed,
            remaining=remaining,
            meals=meal_logs,
            micronutrient_totals=Micronutrients(**agg_micros),
            feedback=feedback,
        )

    # ── Private helpers ───────────────────────────────────────

    def _parse_food_text(self, description: str) -> list[dict[str, Any]]:
        """Extract food items from natural language text."""
        prompt = FOOD_PARSE_PROMPT.format(description=description)
        raw = self._llm.generate(prompt)
        return self._parse_items_json(raw)

    def _parse_items_json(self, raw: str) -> list[dict[str, Any]]:
        """Parse a JSON array of food items from LLM output."""
        import json

        clean = raw.strip()
        # Remove markdown fences
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                p = part.lstrip("json").strip()
                if p.startswith("["):
                    clean = p
                    break
        # Find the JSON array boundaries
        start = clean.find("[")
        end = clean.rfind("]")
        if start != -1 and end != -1:
            clean = clean[start : end + 1]

        try:
            return json.loads(clean)
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Food item parse failed: %s\nRaw: %s", exc, raw[:300])
            return []

    def _build_meal_log(
        self,
        profile: UserProfile,
        raw_items: list[dict[str, Any]],
        source: MealSource,
        raw_input: str,
        meal_type: MealType,
        daily_summary: DailySummary | None,
    ) -> MealLog:
        """
        Enrich raw item list with nutrition data and build a MealLog.
        """
        meal_items: list[MealItem] = []
        for item_dict in raw_items:
            food_name = item_dict.get("food_name", "unknown")
            quantity_g = float(item_dict.get("quantity_g", 100))
            confidence = float(item_dict.get("confidence", 1.0))

            nutrition = self._knowledge.get_nutrition(food_name, quantity_g)
            meal_items.append(
                MealItem(
                    food_name=food_name,
                    quantity_g=quantity_g,
                    nutrition=nutrition,
                    confidence=confidence,
                )
            )

        log = MealLog(
            log_id=str(uuid.uuid4()),
            user_id=profile.user_id,
            timestamp=datetime.utcnow(),
            meal_type=meal_type,
            source=source,
            raw_input=raw_input,
            items=meal_items,
        )
        log.compute_totals()

        # Generate feedback if daily context available
        if daily_summary:
            # Update daily summary with this meal
            updated_summary = self._add_meal_to_summary(log, daily_summary)
            log.alerts = self.generate_feedback(profile, log, updated_summary)
        else:
            log.alerts = self._rule_based_alerts(log)

        return log

    def _add_meal_to_summary(
        self, meal_log: MealLog, summary: DailySummary
    ) -> DailySummary:
        """Return an updated DailySummary with the new meal's macros added."""
        new_macros = meal_log.total_macros
        updated_consumed = Macronutrients(
            calories=summary.consumed.calories + new_macros.calories,
            protein_g=summary.consumed.protein_g + new_macros.protein_g,
            carbohydrates_g=summary.consumed.carbohydrates_g + new_macros.carbohydrates_g,
            fat_g=summary.consumed.fat_g + new_macros.fat_g,
            fiber_g=summary.consumed.fiber_g + new_macros.fiber_g,
            sugar_g=summary.consumed.sugar_g + new_macros.sugar_g,
            saturated_fat_g=summary.consumed.saturated_fat_g + new_macros.saturated_fat_g,
            trans_fat_g=summary.consumed.trans_fat_g + new_macros.trans_fat_g,
        )
        summary.consumed = updated_consumed
        return summary

    def _rule_based_alerts(self, meal_log: MealLog) -> list[str]:
        """Fast rule-based alerts without LLM call."""
        alerts: list[str] = []
        m = meal_log.total_macros
        if m.calories > 900:
            alerts.append(f"⚠️ High-calorie meal ({m.calories:.0f} kcal). Consider smaller portions.")
        if m.sodium_mg > 800 if hasattr(m, 'sodium_mg') else False:
            alerts.append("⚠️ High sodium in this meal. Balance with low-sodium foods later.")
        if m.protein_g < 10:
            alerts.append("💡 Low protein meal. Consider adding a protein source at your next meal.")
        if m.fiber_g > 10:
            alerts.append("✅ Excellent fiber content in this meal!")
        return alerts

    def _generate_daily_feedback(
        self,
        profile: UserProfile,
        consumed: Macronutrients,
        target: MacroBudget,
    ) -> list[str]:
        """Generate end-of-day summary feedback."""
        feedback: list[str] = []
        pct = (consumed.calories / target.calories * 100) if target.calories else 0

        if pct < 85:
            feedback.append(f"🔵 Calorie intake low today ({pct:.0f}% of target). Ensure adequate fuel.")
        elif pct > 115:
            feedback.append(f"🟡 Slightly over calorie target ({pct:.0f}%). Stay consistent tomorrow.")
        else:
            feedback.append(f"✅ Great calorie balance today! ({pct:.0f}% of target)")

        protein_pct = (consumed.protein_g / target.protein_g * 100) if target.protein_g else 0
        if protein_pct < 80:
            feedback.append(f"💪 Protein intake low ({consumed.protein_g:.0f}g / {target.protein_g:.0f}g). "
                          "Add more lean protein tomorrow.")
        if consumed.fiber_g < 20:
            feedback.append("🥦 Fiber intake below 20g. Add more vegetables, legumes, or whole grains.")

        return feedback
