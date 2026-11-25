"""
src/models/nutrition.py
Pydantic v2 schemas for nutritional data.
"""
from __future__ import annotations

from pydantic import BaseModel, Field


class Macronutrients(BaseModel):
    calories: float = Field(0.0, ge=0, description="kcal")
    protein_g: float = Field(0.0, ge=0, description="Protein in grams")
    carbohydrates_g: float = Field(0.0, ge=0, description="Total carbs in grams")
    fat_g: float = Field(0.0, ge=0, description="Total fat in grams")
    fiber_g: float = Field(0.0, ge=0, description="Dietary fiber in grams")
    sugar_g: float = Field(0.0, ge=0, description="Total sugars in grams")
    saturated_fat_g: float = Field(0.0, ge=0)
    trans_fat_g: float = Field(0.0, ge=0)


class Micronutrients(BaseModel):
    # Vitamins
    vitamin_a_mcg: float = Field(0.0, ge=0)
    vitamin_c_mg: float = Field(0.0, ge=0)
    vitamin_d_mcg: float = Field(0.0, ge=0)
    vitamin_e_mg: float = Field(0.0, ge=0)
    vitamin_k_mcg: float = Field(0.0, ge=0)
    vitamin_b12_mcg: float = Field(0.0, ge=0)
    folate_mcg: float = Field(0.0, ge=0)
    # Minerals
    calcium_mg: float = Field(0.0, ge=0)
    iron_mg: float = Field(0.0, ge=0)
    magnesium_mg: float = Field(0.0, ge=0)
    potassium_mg: float = Field(0.0, ge=0)
    sodium_mg: float = Field(0.0, ge=0)
    zinc_mg: float = Field(0.0, ge=0)
    selenium_mcg: float = Field(0.0, ge=0)
    phosphorus_mg: float = Field(0.0, ge=0)


class NutritionData(BaseModel):
    food_name: str
    serving_size_g: float = Field(100.0, gt=0)
    macros: Macronutrients = Field(default_factory=Macronutrients)
    micros: Micronutrients = Field(default_factory=Micronutrients)
    glycemic_index: float | None = Field(None, ge=0, le=100)
    glycemic_load: float | None = Field(None, ge=0)
    allergens: list[str] = Field(default_factory=list)
    source: str = "USDA FoodData Central"

    def scale_to(self, target_g: float) -> "NutritionData":
        """Return a new NutritionData scaled to target_g grams."""
        factor = target_g / self.serving_size_g
        scaled_macros = Macronutrients(
            **{k: v * factor for k, v in self.macros.model_dump().items()}
        )
        scaled_micros = Micronutrients(
            **{k: v * factor for k, v in self.micros.model_dump().items()}
        )
        return NutritionData(
            food_name=self.food_name,
            serving_size_g=target_g,
            macros=scaled_macros,
            micros=scaled_micros,
            glycemic_index=self.glycemic_index,
            glycemic_load=self.glycemic_load,
            allergens=self.allergens,
            source=self.source,
        )
