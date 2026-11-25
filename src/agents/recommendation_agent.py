"""
src/agents/recommendation_agent.py
─────────────────────────────────────────────────────────────
Diet Recommendation Agent.
Generates personalised 7-day meal plans using:
  - User profile (BMR/TDEE, dietary philosophy, fitness goal)
  - RAG-retrieved food options
  - Granite 3 for plan generation
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.guardrails import GuardrailsFilter
from src.models.user import UserProfile
from src.rag.retriever import FoodRetriever
from src.services.watsonx_service import get_llm_client

logger = logging.getLogger(__name__)

MEAL_PLAN_PROMPT = """\
<|system|>
You are an expert registered dietitian AI. Generate a detailed, personalised 7-day meal plan.
Return ONLY valid JSON with no additional commentary.

<|user|>
USER PROFILE:
- Name: {name}
- Age: {age} | Sex: {sex}
- Weight: {weight_kg} kg | Height: {height_cm} cm
- Activity Level: {activity_level}
- Dietary Philosophy: {dietary_philosophy}
- Fitness Goal: {fitness_goal}
- Health Conditions: {health_conditions}
- Allergies: {allergies}
- Daily Calorie Target: {calorie_target:.0f} kcal
- Protein Target: {protein_target:.0f}g | Carbs: {carbs_target:.0f}g | Fat: {fat_target:.0f}g

AVAILABLE FOODS FROM DATABASE (use these as primary ingredients):
{food_context}

Generate a 7-day meal plan. For each day provide breakfast, lunch, dinner, and 1-2 snacks.
Each meal must specify:
- meal_name: descriptive name
- meal_type: breakfast/lunch/dinner/snack
- items: list of {{food_name, quantity_g, description}}
- estimated_calories: number
- notes: any preparation tips or substitution options

Return this JSON structure:
{{
  "plan_name": "string",
  "daily_calorie_target": number,
  "weekly_summary": "string",
  "days": [
    {{
      "day": 1,
      "day_name": "Monday",
      "total_calories": number,
      "meals": [
        {{
          "meal_type": "breakfast",
          "meal_name": "string",
          "items": [{{"food_name": "string", "quantity_g": number, "description": "string"}}],
          "estimated_calories": number,
          "notes": "string"
        }}
      ]
    }}
  ],
  "grocery_list": {{"category": ["item1", "item2"]}},
  "substitution_options": ["list of substitution suggestions"]
}}
<|assistant|>
"""


class RecommendationAgent:
    """
    Generates personalised meal plans and dietary recommendations
    based on user profile and goals.
    """

    def __init__(self) -> None:
        self._retriever = FoodRetriever()
        self._llm = get_llm_client()
        self._guardrails = GuardrailsFilter(use_llm_guardian=True)

    def generate_meal_plan(self, profile: UserProfile) -> dict[str, Any]:
        """
        Generate a 7-day meal plan for the given user profile.

        Returns:
            dict containing the structured meal plan.
        """
        import json

        # 1. Calculate macro targets
        calorie_target = profile.calorie_target
        protein_target = (calorie_target * 0.25) / 4   # 25% from protein
        fat_target = (calorie_target * 0.30) / 9       # 30% from fat
        carbs_target = (calorie_target * 0.45) / 4     # 45% from carbs

        # 2. Retrieve relevant food options matching dietary philosophy
        query = (
            f"{profile.dietary_philosophy.value} foods "
            f"for {profile.fitness_goal.value} "
            f"high protein balanced diet"
        )
        chunks = self._retriever.filter_by_allergen_free(
            query_text=query,
            excluded_allergens=profile.allergies,
            n_results=10,
        )
        food_context = "\n".join(c.text[:200] for c in chunks) if chunks else "General whole foods"

        # 3. Build and execute prompt
        prompt = MEAL_PLAN_PROMPT.format(
            name=profile.name,
            age=profile.age,
            sex=profile.sex,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            activity_level=profile.activity_level.value,
            dietary_philosophy=profile.dietary_philosophy.value,
            fitness_goal=profile.fitness_goal.value,
            health_conditions=", ".join(c.value for c in profile.health_conditions) or "None",
            allergies=", ".join(profile.allergies) or "None",
            calorie_target=calorie_target,
            protein_target=protein_target,
            carbs_target=carbs_target,
            fat_target=fat_target,
            food_context=food_context,
        )

        raw = self._guardrails.safe_generate(
            lambda: self._llm.generate(prompt),
            is_health_content=True,
        )

        # 4. Parse JSON response
        return self._parse_meal_plan(raw, profile, calorie_target)

    def get_substitutions(
        self, food_name: str, reason: str, profile: UserProfile
    ) -> list[str]:
        """
        Suggest ingredient substitutions based on dietary constraints.

        Args:
            food_name: The food to substitute
            reason: Reason for substitution (allergy, preference, availability)
            profile: User profile for context

        Returns:
            List of substitution suggestions.
        """
        prompt = (
            f"<|system|>\nYou are a nutrition expert. Suggest 5 healthy substitutes "
            f"for '{food_name}' for a {profile.dietary_philosophy.value} "
            f"{profile.fitness_goal.value} diet. Reason: {reason}. "
            f"Allergies to avoid: {', '.join(profile.allergies) or 'none'}. "
            "Return a JSON array of strings.\n<|assistant|>"
        )
        raw = self._llm.generate(prompt)
        try:
            import json
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return [raw.strip()]

    def calculate_macros_summary(self, profile: UserProfile) -> dict[str, float]:
        """Return the target macro breakdown dict for a user profile."""
        cal = profile.calorie_target
        return {
            "calories": round(cal, 1),
            "protein_g": round((cal * 0.25) / 4, 1),
            "carbohydrates_g": round((cal * 0.45) / 4, 1),
            "fat_g": round((cal * 0.30) / 9, 1),
            "fiber_g": 30.0,
            "bmr": round(profile.bmr, 1),
            "tdee": round(profile.tdee, 1),
        }

    # ── Private helpers ───────────────────────────────────────

    def _parse_meal_plan(
        self, raw: str, profile: UserProfile, calorie_target: float
    ) -> dict[str, Any]:
        import json

        clean = raw.strip()
        # Remove markdown fences
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                p = part.lstrip("json").strip()
                if p.startswith("{"):
                    clean = p
                    break

        try:
            plan = json.loads(clean)
            plan["user_id"] = profile.user_id
            plan["macro_targets"] = self.calculate_macros_summary(profile)
            return plan
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Meal plan JSON parse failed: %s", exc)
            return {
                "user_id": profile.user_id,
                "plan_name": f"7-Day {profile.dietary_philosophy.value.title()} Plan",
                "daily_calorie_target": round(calorie_target),
                "macro_targets": self.calculate_macros_summary(profile),
                "raw_plan": raw,
                "parse_error": str(exc),
            }
