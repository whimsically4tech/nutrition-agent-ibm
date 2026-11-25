"""
src/models/user.py
Pydantic v2 schemas for user profiles.
"""
from __future__ import annotations

from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, field_validator


class ActivityLevel(str, Enum):
    SEDENTARY = "sedentary"          # desk job, no exercise
    LIGHTLY_ACTIVE = "lightly_active"  # 1-3 days/week
    MODERATELY_ACTIVE = "moderately_active"  # 3-5 days/week
    VERY_ACTIVE = "very_active"       # 6-7 days/week
    SUPER_ACTIVE = "super_active"     # physical job + hard exercise


class DietaryPhilosophy(str, Enum):
    OMNIVORE = "omnivore"
    VEGETARIAN = "vegetarian"
    VEGAN = "vegan"
    PESCATARIAN = "pescatarian"
    KETO = "keto"
    PALEO = "paleo"
    MEDITERRANEAN = "mediterranean"
    HALAL = "halal"
    KOSHER = "kosher"
    JAIN = "jain"
    GLUTEN_FREE = "gluten_free"
    LOW_FODMAP = "low_fodmap"


class FitnessGoal(str, Enum):
    FAT_LOSS = "fat_loss"
    MUSCLE_GAIN = "muscle_gain"
    MAINTENANCE = "maintenance"
    ATHLETIC_PERFORMANCE = "athletic_performance"
    GENERAL_WELLNESS = "general_wellness"


class HealthCondition(str, Enum):
    NONE = "none"
    TYPE2_DIABETES = "type2_diabetes"
    HYPERTENSION = "hypertension"
    CKD = "chronic_kidney_disease"
    HIGH_CHOLESTEROL = "high_cholesterol"
    IBS = "irritable_bowel_syndrome"
    CELIAC = "celiac_disease"
    GOUT = "gout"
    PCOS = "pcos"


class UserProfile(BaseModel):
    user_id: str = Field(..., description="Unique user identifier")
    name: str
    age: Annotated[int, Field(ge=10, le=120)]
    sex: str = Field(..., pattern="^(male|female|other)$")
    weight_kg: Annotated[float, Field(gt=10, lt=500)]
    height_cm: Annotated[float, Field(gt=50, lt=300)]
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    dietary_philosophy: DietaryPhilosophy = DietaryPhilosophy.OMNIVORE
    fitness_goal: FitnessGoal = FitnessGoal.MAINTENANCE
    health_conditions: list[HealthCondition] = Field(default_factory=list)
    allergies: list[str] = Field(default_factory=list)
    medications: list[str] = Field(default_factory=list, description="For drug-nutrient checks")
    daily_calorie_target: float | None = None  # auto-calculated if None

    @field_validator("sex", mode="before")
    @classmethod
    def _normalise_sex(cls, v: str) -> str:
        return v.lower() if isinstance(v, str) else v

    @property
    def bmr(self) -> float:
        """Mifflin-St Jeor BMR."""
        if self.sex == "male":
            return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age + 5
        return 10 * self.weight_kg + 6.25 * self.height_cm - 5 * self.age - 161

    @property
    def tdee(self) -> float:
        """Total Daily Energy Expenditure."""
        multipliers = {
            ActivityLevel.SEDENTARY: 1.2,
            ActivityLevel.LIGHTLY_ACTIVE: 1.375,
            ActivityLevel.MODERATELY_ACTIVE: 1.55,
            ActivityLevel.VERY_ACTIVE: 1.725,
            ActivityLevel.SUPER_ACTIVE: 1.9,
        }
        return self.bmr * multipliers[self.activity_level]

    @property
    def calorie_target(self) -> float:
        if self.daily_calorie_target:
            return self.daily_calorie_target
        adjustments = {
            FitnessGoal.FAT_LOSS: -500,
            FitnessGoal.MUSCLE_GAIN: +300,
            FitnessGoal.MAINTENANCE: 0,
            FitnessGoal.ATHLETIC_PERFORMANCE: +400,
            FitnessGoal.GENERAL_WELLNESS: 0,
        }
        return self.tdee + adjustments[self.fitness_goal]
