"""
src/models/meal.py
Pydantic v2 schemas for meal logging and daily summaries.
"""
from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

from .nutrition import Macronutrients, Micronutrients, NutritionData


class MealSource(str, Enum):
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"


class MealType(str, Enum):
    BREAKFAST = "breakfast"
    LUNCH = "lunch"
    DINNER = "dinner"
    SNACK = "snack"


class MealItem(BaseModel):
    food_name: str
    quantity_g: float = Field(gt=0, description="Portion weight in grams")
    nutrition: NutritionData | None = None
    confidence: float = Field(1.0, ge=0, le=1, description="Detection confidence for image/audio")


class MealLog(BaseModel):
    log_id: str
    user_id: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    meal_type: MealType = MealType.SNACK
    source: MealSource = MealSource.TEXT
    raw_input: str = Field("", description="Original text / transcript / image path")
    items: list[MealItem] = Field(default_factory=list)
    total_macros: Macronutrients = Field(default_factory=Macronutrients)
    notes: str = ""
    alerts: list[str] = Field(default_factory=list, description="Real-time feedback alerts")

    def compute_totals(self) -> None:
        """Aggregate macros across all items in-place."""
        agg: dict[str, float] = {k: 0.0 for k in Macronutrients.model_fields}
        for item in self.items:
            if item.nutrition:
                for k, v in item.nutrition.macros.model_dump().items():
                    agg[k] += v
        self.total_macros = Macronutrients(**agg)


class MacroBudget(BaseModel):
    calories: float
    protein_g: float
    carbohydrates_g: float
    fat_g: float
    fiber_g: float


class DailySummary(BaseModel):
    user_id: str
    date: str  # ISO 8601 date string
    target: MacroBudget
    consumed: Macronutrients = Field(default_factory=Macronutrients)
    remaining: MacroBudget | None = None
    meals: list[MealLog] = Field(default_factory=list)
    micronutrient_totals: Micronutrients = Field(default_factory=Micronutrients)
    feedback: list[str] = Field(default_factory=list)
    streak_days: int = 0
