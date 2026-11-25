"""
tests/test_agents.py
─────────────────────────────────────────────────────────────
Unit tests for all four specialist agents and the orchestrator.
Uses mocking to avoid real IBM watsonx.ai API calls.
"""
from __future__ import annotations

import json
import uuid
from unittest.mock import MagicMock, patch

import pytest

from src.models.nutrition import Macronutrients, Micronutrients, NutritionData
from src.models.user import (
    ActivityLevel,
    DietaryPhilosophy,
    FitnessGoal,
    HealthCondition,
    UserProfile,
)
from src.models.meal import MealItem, MealLog, MealSource, MealType


# ─────────────────────────────────────────────────────────────
#  Fixtures
# ─────────────────────────────────────────────────────────────

@pytest.fixture
def sample_profile() -> UserProfile:
    return UserProfile(
        user_id="test-user-001",
        name="Test User",
        age=30,
        sex="male",
        weight_kg=75.0,
        height_cm=175.0,
        activity_level=ActivityLevel.MODERATELY_ACTIVE,
        dietary_philosophy=DietaryPhilosophy.OMNIVORE,
        fitness_goal=FitnessGoal.MAINTENANCE,
        health_conditions=[],
        allergies=[],
        medications=[],
    )


@pytest.fixture
def diabetic_profile() -> UserProfile:
    return UserProfile(
        user_id="test-user-002",
        name="Diabetic User",
        age=55,
        sex="female",
        weight_kg=80.0,
        height_cm=160.0,
        activity_level=ActivityLevel.LIGHTLY_ACTIVE,
        dietary_philosophy=DietaryPhilosophy.OMNIVORE,
        fitness_goal=FitnessGoal.FAT_LOSS,
        health_conditions=[HealthCondition.TYPE2_DIABETES, HealthCondition.HYPERTENSION],
        allergies=["nuts"],
        medications=["metformin", "lisinopril"],
    )


@pytest.fixture
def sample_nutrition() -> NutritionData:
    return NutritionData(
        food_name="Chicken Breast",
        serving_size_g=150.0,
        macros=Macronutrients(
            calories=180.0,
            protein_g=33.75,
            carbohydrates_g=0.0,
            fat_g=3.93,
            fiber_g=0.0,
        ),
        micros=Micronutrients(
            vitamin_b12_mcg=0.45,
            selenium_mcg=41.4,
            phosphorus_mg=330,
        ),
        glycemic_index=0,
        allergens=[],
    )


@pytest.fixture
def sample_meal_log(sample_profile, sample_nutrition) -> MealLog:
    log = MealLog(
        log_id=str(uuid.uuid4()),
        user_id=sample_profile.user_id,
        meal_type=MealType.LUNCH,
        source=MealSource.TEXT,
        raw_input="150g grilled chicken breast",
        items=[
            MealItem(
                food_name="Chicken Breast",
                quantity_g=150.0,
                nutrition=sample_nutrition,
                confidence=1.0,
            )
        ],
    )
    log.compute_totals()
    return log


# ─────────────────────────────────────────────────────────────
#  Test: UserProfile model
# ─────────────────────────────────────────────────────────────

class TestUserProfile:
    def test_bmr_male(self, sample_profile):
        """Mifflin-St Jeor BMR for male."""
        expected = 10 * 75 + 6.25 * 175 - 5 * 30 + 5
        assert abs(sample_profile.bmr - expected) < 0.01

    def test_bmr_female(self):
        p = UserProfile(
            user_id="u", name="F", age=25, sex="female",
            weight_kg=60.0, height_cm=165.0,
        )
        expected = 10 * 60 + 6.25 * 165 - 5 * 25 - 161
        assert abs(p.bmr - expected) < 0.01

    def test_tdee_moderately_active(self, sample_profile):
        expected_tdee = sample_profile.bmr * 1.55
        assert abs(sample_profile.tdee - expected_tdee) < 0.01

    def test_calorie_target_fat_loss(self):
        p = UserProfile(
            user_id="u", name="T", age=30, sex="male",
            weight_kg=85.0, height_cm=180.0,
            fitness_goal=FitnessGoal.FAT_LOSS,
        )
        assert p.calorie_target == p.tdee - 500

    def test_calorie_target_muscle_gain(self):
        p = UserProfile(
            user_id="u", name="T", age=30, sex="male",
            weight_kg=70.0, height_cm=175.0,
            fitness_goal=FitnessGoal.MUSCLE_GAIN,
        )
        assert p.calorie_target == p.tdee + 300

    def test_sex_normalised_to_lowercase(self):
        p = UserProfile(user_id="u", name="T", age=30, sex="Male", weight_kg=70, height_cm=175)
        assert p.sex == "male"


# ─────────────────────────────────────────────────────────────
#  Test: NutritionData model
# ─────────────────────────────────────────────────────────────

class TestNutritionData:
    def test_scale_to_doubles_calories(self, sample_nutrition):
        scaled = sample_nutrition.scale_to(300.0)
        assert abs(scaled.macros.calories - sample_nutrition.macros.calories * 2) < 0.1

    def test_scale_to_maintains_proportions(self, sample_nutrition):
        scaled = sample_nutrition.scale_to(75.0)
        factor = 75.0 / 150.0
        assert abs(scaled.macros.protein_g - sample_nutrition.macros.protein_g * factor) < 0.01

    def test_scale_to_updates_serving_size(self, sample_nutrition):
        scaled = sample_nutrition.scale_to(200.0)
        assert scaled.serving_size_g == 200.0


# ─────────────────────────────────────────────────────────────
#  Test: MealLog model
# ─────────────────────────────────────────────────────────────

class TestMealLog:
    def test_compute_totals_sums_calories(self, sample_meal_log, sample_nutrition):
        assert abs(sample_meal_log.total_macros.calories - sample_nutrition.macros.calories) < 0.1

    def test_compute_totals_sums_protein(self, sample_meal_log, sample_nutrition):
        assert abs(sample_meal_log.total_macros.protein_g - sample_nutrition.macros.protein_g) < 0.01

    def test_multiple_items_aggregated(self, sample_profile, sample_nutrition):
        item1 = MealItem(food_name="Food1", quantity_g=100, nutrition=sample_nutrition)
        item2 = MealItem(food_name="Food2", quantity_g=100, nutrition=sample_nutrition)
        log = MealLog(
            log_id="x", user_id=sample_profile.user_id,
            items=[item1, item2]
        )
        log.compute_totals()
        assert abs(log.total_macros.calories - sample_nutrition.macros.calories * 2) < 0.1


# ─────────────────────────────────────────────────────────────
#  Test: KnowledgeAgent (mocked LLM)
# ─────────────────────────────────────────────────────────────

MOCK_NUTRITION_JSON = json.dumps({
    "food_name": "Brown Rice",
    "serving_size_g": 180.0,
    "macros": {
        "calories": 216.0, "protein_g": 5.0, "carbohydrates_g": 45.0,
        "fat_g": 1.8, "fiber_g": 3.5, "sugar_g": 0.7,
        "saturated_fat_g": 0.4, "trans_fat_g": 0.0
    },
    "micros": {
        "vitamin_a_mcg": 0, "vitamin_c_mg": 0, "vitamin_d_mcg": 0,
        "vitamin_e_mg": 0.5, "vitamin_k_mcg": 1.0, "vitamin_b12_mcg": 0,
        "folate_mcg": 7, "calcium_mg": 10, "iron_mg": 0.6,
        "magnesium_mg": 42, "potassium_mg": 77, "sodium_mg": 1,
        "zinc_mg": 0.6, "selenium_mcg": 14, "phosphorus_mg": 80
    },
    "glycemic_index": 55,
    "allergens": []
})


class TestKnowledgeAgent:
    @patch("src.agents.knowledge_agent.get_llm_client")
    @patch("src.agents.knowledge_agent.FoodRetriever")
    def test_get_nutrition_returns_nutrition_data(self, mock_retriever_cls, mock_llm_cls):
        from src.agents.knowledge_agent import KnowledgeAgent
        from src.rag.retriever import RetrievedChunk

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MOCK_NUTRITION_JSON
        mock_llm_cls.return_value = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = [
            RetrievedChunk(id="USDA-20035", text="Brown rice nutrition info", metadata={})
        ]
        mock_retriever_cls.return_value = mock_retriever

        agent = KnowledgeAgent()
        result = agent.get_nutrition("brown rice", 180)

        assert isinstance(result, NutritionData)
        assert result.food_name == "Brown Rice"
        assert result.macros.calories == 216.0
        assert result.macros.protein_g == 5.0

    @patch("src.agents.knowledge_agent.get_llm_client")
    @patch("src.agents.knowledge_agent.FoodRetriever")
    def test_get_nutrition_fallback_on_no_results(self, mock_retriever_cls, mock_llm_cls):
        from src.agents.knowledge_agent import KnowledgeAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MOCK_NUTRITION_JSON
        mock_llm_cls.return_value = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.query.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        agent = KnowledgeAgent()
        result = agent.get_nutrition("unknown exotic food", 100)
        assert isinstance(result, NutritionData)


# ─────────────────────────────────────────────────────────────
#  Test: RecommendationAgent (mocked)
# ─────────────────────────────────────────────────────────────

MOCK_MEAL_PLAN_JSON = json.dumps({
    "plan_name": "7-Day Omnivore Maintenance Plan",
    "daily_calorie_target": 2300,
    "weekly_summary": "Balanced whole foods plan.",
    "days": [
        {
            "day": 1,
            "day_name": "Monday",
            "total_calories": 2290,
            "meals": [
                {
                    "meal_type": "breakfast",
                    "meal_name": "Oats and Eggs",
                    "items": [{"food_name": "oats", "quantity_g": 80, "description": "cooked"}],
                    "estimated_calories": 300,
                    "notes": "Add berries for antioxidants."
                }
            ]
        }
    ],
    "grocery_list": {"grains": ["oats", "brown rice"]},
    "substitution_options": ["Swap oats for quinoa for more protein."]
})


class TestRecommendationAgent:
    @patch("src.agents.recommendation_agent.get_llm_client")
    @patch("src.agents.recommendation_agent.FoodRetriever")
    @patch("src.agents.recommendation_agent.GuardrailsFilter")
    def test_generate_meal_plan_returns_dict(
        self, mock_guard_cls, mock_retriever_cls, mock_llm_cls, sample_profile
    ):
        from src.agents.recommendation_agent import RecommendationAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = MOCK_MEAL_PLAN_JSON
        mock_llm_cls.return_value = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.filter_by_allergen_free.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        mock_guard = MagicMock()
        mock_guard.safe_generate.return_value = MOCK_MEAL_PLAN_JSON
        mock_guard_cls.return_value = mock_guard

        agent = RecommendationAgent()
        plan = agent.generate_meal_plan(sample_profile)

        assert isinstance(plan, dict)
        assert plan["user_id"] == sample_profile.user_id

    def test_calculate_macros_summary_calorie_total(self, sample_profile):
        from src.agents.recommendation_agent import RecommendationAgent
        with patch("src.agents.recommendation_agent.FoodRetriever"), \
             patch("src.agents.recommendation_agent.get_llm_client"), \
             patch("src.agents.recommendation_agent.GuardrailsFilter"):
            agent = RecommendationAgent()
            macros = agent.calculate_macros_summary(sample_profile)

        assert macros["calories"] == pytest.approx(sample_profile.calorie_target, rel=0.01)
        # Protein kcal + Carbs kcal + Fat kcal ≈ total calories (within 1g rounding)
        cal_from_macros = macros["protein_g"] * 4 + macros["carbohydrates_g"] * 4 + macros["fat_g"] * 9
        assert abs(cal_from_macros - macros["calories"]) < 5


# ─────────────────────────────────────────────────────────────
#  Test: HealthAdvisoryAgent (mocked)
# ─────────────────────────────────────────────────────────────

class TestHealthAdvisoryAgent:
    @patch("src.agents.health_agent.get_llm_client")
    @patch("src.agents.health_agent.GuidelineRetriever")
    @patch("src.agents.health_agent.GuardrailsFilter")
    def test_get_health_advice_returns_string(
        self, mock_guard_cls, mock_retriever_cls, mock_llm_cls, diabetic_profile
    ):
        from src.agents.health_agent import HealthAdvisoryAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = "Avoid high-GI foods. Prioritise vegetables and legumes."
        mock_llm_cls.return_value = mock_llm

        mock_retriever = MagicMock()
        mock_retriever.get_condition_guidelines.return_value = None
        mock_retriever.query.return_value = []
        mock_retriever_cls.return_value = mock_retriever

        mock_guard = MagicMock()
        mock_guard.screen_input.return_value = (True, "OK")
        mock_guard.safe_generate.return_value = "Safe advice response."
        mock_guard_cls.return_value = mock_guard

        agent = HealthAdvisoryAgent()
        advice = agent.get_health_advice(diabetic_profile)

        assert isinstance(advice, str)
        assert len(advice) > 0

    @patch("src.agents.health_agent.get_llm_client")
    @patch("src.agents.health_agent.GuidelineRetriever")
    @patch("src.agents.health_agent.GuardrailsFilter")
    def test_harmful_input_blocked(
        self, mock_guard_cls, mock_retriever_cls, mock_llm_cls, diabetic_profile
    ):
        from src.agents.health_agent import HealthAdvisoryAgent

        mock_llm_cls.return_value = MagicMock()
        mock_retriever_cls.return_value = MagicMock()

        mock_guard = MagicMock()
        mock_guard.screen_input.return_value = (False, "Dangerous medical claim detected.")
        mock_guard_cls.return_value = mock_guard

        agent = HealthAdvisoryAgent()
        result = agent.get_health_advice(diabetic_profile, "Cure my diabetes with this diet")

        assert "could not be processed" in result.lower()


# ─────────────────────────────────────────────────────────────
#  Test: FoodLogAgent (mocked)
# ─────────────────────────────────────────────────────────────

class TestFoodLogAgent:
    @patch("src.agents.food_log_agent.get_llm_client")
    @patch("src.agents.food_log_agent.get_vision_client")
    @patch("src.agents.food_log_agent.SpeechToTextService")
    @patch("src.agents.food_log_agent.KnowledgeAgent")
    def test_log_meal_text_returns_meal_log(
        self, mock_ka_cls, mock_stt_cls, mock_vision_cls, mock_llm_cls, sample_profile, sample_nutrition
    ):
        from src.agents.food_log_agent import FoodLogAgent

        mock_llm = MagicMock()
        mock_llm.generate.return_value = '[{"food_name": "scrambled eggs", "quantity_g": 120, "confidence": 1.0}]'
        mock_llm_cls.return_value = mock_llm

        mock_ka = MagicMock()
        mock_ka.get_nutrition.return_value = sample_nutrition
        mock_ka_cls.return_value = mock_ka

        mock_stt_cls.return_value = MagicMock()
        mock_vision_cls.return_value = MagicMock()

        agent = FoodLogAgent()
        log = agent.log_meal_text(sample_profile, "2 scrambled eggs", MealType.BREAKFAST)

        assert isinstance(log, MealLog)
        assert log.source == MealSource.TEXT
        assert log.meal_type == MealType.BREAKFAST
        assert len(log.items) >= 1
        assert log.total_macros.calories > 0

    @patch("src.agents.food_log_agent.get_llm_client")
    @patch("src.agents.food_log_agent.get_vision_client")
    @patch("src.agents.food_log_agent.SpeechToTextService")
    @patch("src.agents.food_log_agent.KnowledgeAgent")
    def test_log_meal_generates_alerts(
        self, mock_ka_cls, mock_stt_cls, mock_vision_cls, mock_llm_cls, sample_profile, sample_nutrition
    ):
        from src.agents.food_log_agent import FoodLogAgent

        mock_llm = MagicMock()
        # First call: food parsing; second call: feedback generation
        mock_llm.generate.side_effect = [
            '[{"food_name": "burger", "quantity_g": 400, "confidence": 1.0}]',
            "1. High calorie meal. 2. Watch sodium. 3. Add vegetables.",
        ]
        mock_llm_cls.return_value = mock_llm

        # High-calorie nutrition mock
        heavy_nutrition = NutritionData(
            food_name="Burger",
            serving_size_g=400,
            macros=Macronutrients(calories=1200, protein_g=40, carbohydrates_g=80, fat_g=60),
        )
        mock_ka = MagicMock()
        mock_ka.get_nutrition.return_value = heavy_nutrition
        mock_ka_cls.return_value = mock_ka
        mock_stt_cls.return_value = MagicMock()
        mock_vision_cls.return_value = MagicMock()

        agent = FoodLogAgent()
        log = agent.log_meal_text(sample_profile, "large burger with fries")

        # Rule-based alerts should fire for >900 kcal
        assert any("High" in a or "⚠️" in a for a in log.alerts)


# ─────────────────────────────────────────────────────────────
#  Test: Guardrails
# ─────────────────────────────────────────────────────────────

class TestGuardrails:
    def test_regex_blocks_harmful_input(self):
        from src.agents.guardrails import GuardrailsFilter
        guard = GuardrailsFilter(use_llm_guardian=False)
        is_safe, reason = guard.screen_input("Can this diet cure my diabetes?")
        assert not is_safe

    def test_safe_input_passes(self):
        from src.agents.guardrails import GuardrailsFilter
        guard = GuardrailsFilter(use_llm_guardian=False)
        is_safe, _ = guard.screen_input("What vegetables should I eat for more fiber?")
        assert is_safe

    def test_disclaimer_injected_for_health_content(self):
        from src.agents.guardrails import GuardrailsFilter, CLINICAL_DISCLAIMER
        guard = GuardrailsFilter(use_llm_guardian=False)
        output = guard.screen_output("Eat more vegetables.", is_health_content=True)
        assert CLINICAL_DISCLAIMER in output

    def test_disclaimer_not_duplicated(self):
        from src.agents.guardrails import GuardrailsFilter, CLINICAL_DISCLAIMER
        guard = GuardrailsFilter(use_llm_guardian=False)
        text = "Advice." + CLINICAL_DISCLAIMER
        output = guard.screen_output(text, is_health_content=True)
        assert output.count(CLINICAL_DISCLAIMER) == 1
