"""
src/api/main.py
─────────────────────────────────────────────────────────────
FastAPI application exposing the nutrition agent platform.

Endpoints:
  POST /api/v1/recommend-plan     → 7-day personalised meal plan
  POST /api/v1/log-meal           → Log a meal (text/audio/image)
  POST /api/v1/analyze-image      → Analyse a plate photo
  POST /api/v1/health-advice      → Preventive health nutrition advice
  GET  /api/v1/daily-summary      → Today's macro progress and feedback
  POST /api/v1/nutrition-lookup   → Nutrition data for a food item
  GET  /health                    → Health check
"""
from __future__ import annotations

import logging
import uuid
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.agents.food_log_agent import FoodLogAgent
from src.agents.health_agent import HealthAdvisoryAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.orchestrator import NutritionOrchestrator
from src.agents.recommendation_agent import RecommendationAgent
from src.models.meal import DailySummary, MealLog, MealType
from src.models.user import (
    ActivityLevel,
    DietaryPhilosophy,
    FitnessGoal,
    HealthCondition,
    UserProfile,
)

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
#  App initialisation
# ─────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Nutrition Agent API",
    description=(
        "Production-grade multi-agent nutrition platform powered by "
        "IBM Granite 3 and watsonx.ai"
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────────────────────
#  Request / Response schemas
# ─────────────────────────────────────────────────────────────

class UserProfileRequest(BaseModel):
    user_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    age: int = Field(ge=10, le=120)
    sex: str = Field(pattern="^(male|female|other)$")
    weight_kg: float = Field(gt=10, lt=500)
    height_cm: float = Field(gt=50, lt=300)
    activity_level: ActivityLevel = ActivityLevel.MODERATELY_ACTIVE
    dietary_philosophy: DietaryPhilosophy = DietaryPhilosophy.OMNIVORE
    fitness_goal: FitnessGoal = FitnessGoal.MAINTENANCE
    health_conditions: list[HealthCondition] = []
    allergies: list[str] = []
    medications: list[str] = []


class MealPlanRequest(BaseModel):
    profile: UserProfileRequest


class MealLogTextRequest(BaseModel):
    profile: UserProfileRequest
    description: str = Field(..., description="Natural language meal description")
    meal_type: MealType = MealType.SNACK
    daily_summary: dict | None = None


class HealthAdviceRequest(BaseModel):
    profile: UserProfileRequest
    question: str = Field(
        "Provide general preventive dietary advice for my health conditions."
    )


class NutritionLookupRequest(BaseModel):
    food_name: str
    portion_g: float = Field(100.0, gt=0)


class DailySummaryRequest(BaseModel):
    profile: UserProfileRequest
    meal_logs: list[dict] = []
    date: str | None = None


class OrchestratorRequest(BaseModel):
    profile: UserProfileRequest
    message: str
    extra_data: dict = {}


# ─────────────────────────────────────────────────────────────
#  Routes
# ─────────────────────────────────────────────────────────────

@app.get("/health", tags=["System"])
async def health_check() -> dict:
    return {"status": "healthy", "service": "AI Nutrition Agent"}


@app.post("/api/v1/recommend-plan", tags=["Meal Planning"])
async def recommend_meal_plan(request: MealPlanRequest) -> dict[str, Any]:
    """
    Generate a personalised 7-day meal plan based on user profile.
    Calculates BMR/TDEE and tailors meals to dietary philosophy and fitness goal.
    """
    try:
        profile = UserProfile(**request.profile.model_dump())
        agent = RecommendationAgent()
        plan = agent.generate_meal_plan(profile)
        return {
            "status": "success",
            "macro_summary": agent.calculate_macros_summary(profile),
            "meal_plan": plan,
        }
    except Exception as exc:
        logger.error("Meal plan generation failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/log-meal", tags=["Food Logging"])
async def log_meal_text(request: MealLogTextRequest) -> dict[str, Any]:
    """
    Log a meal from a natural language description.
    Returns parsed food items, nutrition totals, and real-time feedback.
    """
    try:
        profile = UserProfile(**request.profile.model_dump())
        agent = FoodLogAgent()

        daily_summary = None
        if request.daily_summary:
            try:
                daily_summary = DailySummary.model_validate(request.daily_summary)
            except Exception:
                pass

        log = agent.log_meal_text(
            profile=profile,
            description=request.description,
            meal_type=request.meal_type,
            daily_summary=daily_summary,
        )
        return {
            "status": "success",
            "log_id": log.log_id,
            "meal_log": log.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Meal log failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/log-meal/audio", tags=["Food Logging"])
async def log_meal_audio(
    audio_file: UploadFile = File(..., description="Audio recording of the meal"),
    user_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    weight_kg: float = Form(...),
    height_cm: float = Form(...),
    meal_type: str = Form(MealType.SNACK.value),
) -> dict[str, Any]:
    """
    Log a meal from an audio recording.
    Transcribes audio → parses foods → returns nutrition data and feedback.
    """
    try:
        audio_bytes = await audio_file.read()
        content_type = audio_file.content_type or "audio/wav"

        profile = UserProfile(
            user_id=user_id, name=name, age=age, sex=sex,
            weight_kg=weight_kg, height_cm=height_cm,
        )
        agent = FoodLogAgent()
        log = agent.log_meal_audio(
            profile=profile,
            audio_bytes=audio_bytes,
            content_type=content_type,
            meal_type=MealType(meal_type),
        )
        return {
            "status": "success",
            "transcript": log.raw_input,
            "meal_log": log.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Audio meal log failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/analyze-image", tags=["Food Logging"])
async def analyze_meal_image(
    image_file: UploadFile = File(..., description="Plate/meal photograph"),
    user_id: str = Form(...),
    name: str = Form(...),
    age: int = Form(...),
    sex: str = Form(...),
    weight_kg: float = Form(...),
    height_cm: float = Form(...),
    meal_type: str = Form(MealType.SNACK.value),
) -> dict[str, Any]:
    """
    Analyse a plate photo using Granite Vision.
    Identifies food items, estimates portions, and returns nutrition data.
    """
    try:
        image_bytes = await image_file.read()

        profile = UserProfile(
            user_id=user_id, name=name, age=age, sex=sex,
            weight_kg=weight_kg, height_cm=height_cm,
        )
        agent = FoodLogAgent()
        log = agent.log_meal_image(
            profile=profile,
            image_data=image_bytes,
            meal_type=MealType(meal_type),
        )
        return {
            "status": "success",
            "vision_analysis": log.raw_input,
            "meal_log": log.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Image analysis failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/health-advice", tags=["Health Advisory"])
async def get_health_advice(request: HealthAdviceRequest) -> dict[str, Any]:
    """
    Get condition-specific dietary advice.
    Covers Type 2 Diabetes, Hypertension, CKD, High Cholesterol.
    Checks for drug-nutrient interactions. Protected by Granite Guardian.
    """
    try:
        profile = UserProfile(**request.profile.model_dump())
        agent = HealthAdvisoryAgent()
        advice = agent.get_health_advice(profile, request.question)
        return {
            "status": "success",
            "conditions": [c.value for c in profile.health_conditions],
            "advice": advice,
        }
    except Exception as exc:
        logger.error("Health advice failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/nutrition-lookup", tags=["Nutrition Data"])
async def nutrition_lookup(request: NutritionLookupRequest) -> dict[str, Any]:
    """
    Retrieve detailed nutritional data for a food item from the RAG database.
    """
    try:
        agent = KnowledgeAgent()
        nutrition = agent.get_nutrition(request.food_name, request.portion_g)
        return {
            "status": "success",
            "nutrition": nutrition.model_dump(),
        }
    except Exception as exc:
        logger.error("Nutrition lookup failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/daily-summary", tags=["Progress Tracking"])
async def get_daily_summary(request: DailySummaryRequest) -> dict[str, Any]:
    """
    Aggregate all logged meals for a day and return macro progress,
    micronutrient totals, and AI-generated feedback.
    """
    try:
        profile = UserProfile(**request.profile.model_dump())
        agent = FoodLogAgent()

        meal_logs: list[MealLog] = []
        for log_data in request.meal_logs:
            try:
                meal_logs.append(MealLog.model_validate(log_data))
            except Exception as exc:
                logger.warning("Invalid meal log skipped: %s", exc)

        summary = agent.build_daily_summary(profile, meal_logs, request.date)
        return {
            "status": "success",
            "daily_summary": summary.model_dump(mode="json"),
        }
    except Exception as exc:
        logger.error("Daily summary failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.post("/api/v1/chat", tags=["Orchestrator"])
async def chat_with_orchestrator(request: OrchestratorRequest) -> dict[str, Any]:
    """
    Natural language interface to the full multi-agent system.
    The Orchestrator classifies intent and routes to the appropriate agent.
    """
    try:
        profile = UserProfile(**request.profile.model_dump())
        orchestrator = NutritionOrchestrator()
        result = orchestrator.run(
            user_request=request.message,
            profile=profile,
            extra_data=request.extra_data,
        )
        return {"status": "success", **result}
    except Exception as exc:
        logger.error("Orchestrator failed: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
#  Startup / Shutdown lifecycle
# ─────────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event() -> None:
    """Pre-warm connections and log startup."""
    from src.config import get_settings
    cfg = get_settings()
    logger.info(
        "🚀 AI Nutrition Agent API started | env=%s | model=%s",
        cfg.app_env,
        cfg.granite_llm_model,
    )


if __name__ == "__main__":
    import uvicorn
    from src.config import get_settings
    cfg = get_settings()
    uvicorn.run(
        "src.api.main:app",
        host=cfg.api_host,
        port=cfg.api_port,
        reload=cfg.app_env == "development",
        log_level=cfg.log_level.lower(),
    )
