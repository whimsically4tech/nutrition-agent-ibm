"""
src/agents/health_agent.py
─────────────────────────────────────────────────────────────
Health Advisory Agent.
Specialises in preventive nutrition for chronic conditions:
  - Type 2 Diabetes (low-GI/GL)
  - Hypertension (DASH diet)
  - CKD (potassium/phosphorus restriction)
  - High cholesterol (TLC diet)

Checks drug-nutrient interactions and applies Granite-Guardian
safety filters before returning any clinical advice.
"""
from __future__ import annotations

import logging
from typing import Any

from src.agents.guardrails import GuardrailsFilter
from src.models.user import HealthCondition, UserProfile
from src.rag.retriever import GuidelineRetriever
from src.services.watsonx_service import get_llm_client

logger = logging.getLogger(__name__)

HEALTH_ADVICE_PROMPT = """\
<|system|>
You are a clinical nutritionist AI assistant specialising in preventive health and chronic condition management. 
Provide evidence-based, specific, and actionable dietary advice grounded in the clinical guidelines provided.
NEVER diagnose or prescribe medication. Always recommend consulting a healthcare provider.

Clinical Guidelines Context:
{guideline_context}

<|user|>
Patient Profile:
- Age: {age} | Sex: {sex}
- Weight: {weight_kg} kg | Height: {height_cm} cm
- Conditions: {conditions}
- Medications: {medications}
- Current Diet: {dietary_philosophy}
- Allergies: {allergies}

Request: {request}

Provide specific dietary recommendations addressing:
1. Foods to prioritise for their condition(s)
2. Foods and nutrients to limit or avoid
3. Any critical drug-nutrient interactions based on their medications
4. Practical meal timing and portion advice
5. Key micronutrients to monitor

Add a clinical disclaimer at the end.
<|assistant|>
"""

DRUG_INTERACTION_PROMPT = """\
<|system|>
You are a clinical pharmacist AI. Check for drug-nutrient interactions.
<|user|>
Patient medications: {medications}
Recent dietary intake: {food_items}
Clinical guidelines context: {context}

Identify any significant drug-nutrient interactions, rate their severity (critical/moderate/mild),
and provide specific avoidance or timing recommendations.
Return as JSON: {{"interactions": [{{"drug": str, "nutrient": str, "severity": str, "recommendation": str}}]}}
<|assistant|>
"""


class HealthAdvisoryAgent:
    """
    Preventive health nutrition advisor backed by clinical guidelines RAG
    and Granite Guardian safety enforcement.
    """

    # Mapping from HealthCondition enum to guideline keys
    CONDITION_MAP = {
        HealthCondition.TYPE2_DIABETES: "type2_diabetes",
        HealthCondition.HYPERTENSION: "hypertension",
        HealthCondition.CKD: "chronic_kidney_disease",
        HealthCondition.HIGH_CHOLESTEROL: "high_cholesterol",
    }

    def __init__(self) -> None:
        self._retriever = GuidelineRetriever()
        self._llm = get_llm_client()
        self._guardrails = GuardrailsFilter(use_llm_guardian=True)

    def get_health_advice(
        self, profile: UserProfile, request: str = "General preventive dietary advice"
    ) -> str:
        """
        Generate condition-specific dietary advice for a user.

        Args:
            profile: User profile including health conditions and medications
            request: Specific health question or area of concern

        Returns:
            Guardrail-filtered advice string with clinical disclaimer.
        """
        # 1. Screen input
        is_safe, reason = self._guardrails.screen_input(request)
        if not is_safe:
            return (
                f"Your request could not be processed: {reason}\n"
                "Please rephrase your question to focus on general dietary guidance."
            )

        # 2. Retrieve relevant clinical guidelines
        guideline_context = self._get_condition_guidelines(profile)

        # 3. Generate advice
        prompt = HEALTH_ADVICE_PROMPT.format(
            age=profile.age,
            sex=profile.sex,
            weight_kg=profile.weight_kg,
            height_cm=profile.height_cm,
            conditions=", ".join(c.value for c in profile.health_conditions) or "None",
            medications=", ".join(profile.medications) or "None",
            dietary_philosophy=profile.dietary_philosophy.value,
            allergies=", ".join(profile.allergies) or "None",
            request=request,
            guideline_context=guideline_context,
        )

        return self._guardrails.safe_generate(
            lambda: self._llm.generate(prompt),
            is_health_content=True,
        )

    def check_drug_interactions(
        self, profile: UserProfile, food_items: list[str]
    ) -> dict[str, Any]:
        """
        Check for drug-nutrient interactions given medications and recently eaten foods.

        Returns:
            dict with key 'interactions' (list of interaction objects).
        """
        import json

        if not profile.medications:
            return {"interactions": []}

        # Retrieve condition-specific interaction guidelines
        guideline_context = self._get_condition_guidelines(profile)

        prompt = DRUG_INTERACTION_PROMPT.format(
            medications=", ".join(profile.medications),
            food_items=", ".join(food_items),
            context=guideline_context[:1500],
        )
        raw = self._llm.generate(prompt)

        # Parse JSON
        clean = raw.strip()
        if "```" in clean:
            parts = clean.split("```")
            for p in parts:
                p = p.lstrip("json").strip()
                if p.startswith("{"):
                    clean = p
                    break

        try:
            result = json.loads(clean)
            # Apply guardian to critical interactions
            critical = [
                i for i in result.get("interactions", [])
                if i.get("severity") == "critical"
            ]
            if critical:
                logger.warning(
                    "Critical drug-nutrient interaction detected: %s",
                    critical,
                )
            return result
        except (json.JSONDecodeError, ValueError) as exc:
            logger.warning("Drug interaction parse failed: %s", exc)
            return {"interactions": [], "raw_response": raw}

    def get_condition_targets(self, condition: HealthCondition) -> dict[str, Any]:
        """
        Return quantitative nutritional targets for a given health condition.
        Sourced from the clinical guidelines database.
        """
        condition_key = self.CONDITION_MAP.get(condition)
        if not condition_key:
            return {}

        chunk = self._retriever.get_condition_guidelines(condition_key)
        if not chunk:
            return {}

        # Extract targets section from metadata text via LLM
        prompt = (
            f"<|system|>\nExtract the specific numeric nutritional targets from this clinical guideline "
            f"as a flat JSON object. Keys should be nutrient names, values should be numeric limits.\n"
            f"Guideline: {chunk.text[:1000]}\n<|assistant|>"
        )
        raw = self._llm.generate(prompt)
        try:
            import json
            clean = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
            return json.loads(clean)
        except Exception:
            return {"guideline_text": chunk.text[:500]}

    # ── Private helpers ───────────────────────────────────────

    def _get_condition_guidelines(self, profile: UserProfile) -> str:
        """Retrieve and concatenate guideline text for all user conditions."""
        contexts: list[str] = []
        for condition in profile.health_conditions:
            condition_key = self.CONDITION_MAP.get(condition)
            if condition_key:
                chunk = self._retriever.get_condition_guidelines(condition_key)
                if chunk:
                    contexts.append(chunk.text)

        if not contexts:
            # General wellness query
            chunks = self._retriever.query("preventive nutrition healthy diet", n_results=2)
            contexts = [c.text for c in chunks]

        return "\n\n---\n".join(contexts) or "General evidence-based dietary guidelines apply."
