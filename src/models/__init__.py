"""
src/models/__init__.py
Exports all Pydantic v2 domain schemas.
"""
from .nutrition import NutritionData, Micronutrients, Macronutrients
from .user import UserProfile, DietaryPhilosophy, FitnessGoal, ActivityLevel
from .meal import MealLog, MealItem, DailySummary, MealSource

__all__ = [
    "NutritionData",
    "Micronutrients",
    "Macronutrients",
    "UserProfile",
    "DietaryPhilosophy",
    "FitnessGoal",
    "ActivityLevel",
    "MealLog",
    "MealItem",
    "DailySummary",
    "MealSource",
]
