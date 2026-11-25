"""
src/agents/orchestrator.py
─────────────────────────────────────────────────────────────
LangGraph Supervisor / Orchestrator.
Routes user requests to the appropriate specialist agent using
a Supervisor → Sub-agent pattern.

Agents:
  - knowledge_agent  : Nutrition data lookup
  - recommendation_agent : Meal plan generation
  - health_agent     : Clinical/preventive health advice
  - food_log_agent   : Multimodal meal logging & feedback

The Supervisor uses Granite 3 to classify intent and dispatch
to the correct agent, then assembles the final response.
"""
from __future__ import annotations

import logging
from enum import Enum
from typing import Annotated, Any, TypedDict

from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from src.agents.food_log_agent import FoodLogAgent
from src.agents.guardrails import GuardrailsFilter
from src.agents.health_agent import HealthAdvisoryAgent
from src.agents.knowledge_agent import KnowledgeAgent
from src.agents.recommendation_agent import RecommendationAgent
from src.models.meal import DailySummary, MealLog, MealType
from src.models.user import UserProfile
from src.services.watsonx_service import get_llm_client

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────
#  Agent intent classification
# ─────────────────────────────────────────────────────────────

class AgentIntent(str, Enum):
    NUTRITION_LOOKUP = "nutrition_lookup"
    MEAL_PLAN = "meal_plan"
    HEALTH_ADVICE = "health_advice"
    LOG_MEAL = "log_meal"
    DAILY_SUMMARY = "daily_summary"
    GENERAL = "general"


CLASSIFIER_PROMPT = """\
<|system|>
Classify the user's nutrition-related request into exactly one intent category.
Return ONLY the category name, nothing else.

Categories:
- nutrition_lookup: asking about calories, macros, or nutrients in a specific food
- meal_plan: requesting a meal plan, weekly diet plan, or food schedule
- health_advice: asking about diet for a health condition, drug interactions, or preventive health
- log_meal: logging a meal they just ate (text, describes food eaten)
- daily_summary: asking for today's nutrition summary, progress, or remaining budget
- general: anything else nutrition-related

<|user|>
Request: "{request}"

Intent:
<|assistant|>
"""


# ─────────────────────────────────────────────────────────────
#  LangGraph state schema
# ─────────────────────────────────────────────────────────────

class OrchestratorState(TypedDict):
    # Input
    user_request: str
    profile: dict  # serialised UserProfile
    extra_data: dict  # audio bytes, image bytes, meal_type, daily_summary, etc.

    # Routing
    intent: str

    # Outputs
    agent_result: Any
    final_response: str
    error: str | None


# ─────────────────────────────────────────────────────────────
#  Node implementations
# ─────────────────────────────────────────────────────────────

def _node_classify(state: OrchestratorState) -> OrchestratorState:
    """Supervisor node: classify intent using Granite LLM."""
    llm = get_llm_client()
    prompt = CLASSIFIER_PROMPT.format(request=state["user_request"])
    raw = llm.generate(prompt).strip().lower()

    # Map to known enum
    intent_map = {
        "nutrition_lookup": AgentIntent.NUTRITION_LOOKUP,
        "meal_plan": AgentIntent.MEAL_PLAN,
        "health_advice": AgentIntent.HEALTH_ADVICE,
        "log_meal": AgentIntent.LOG_MEAL,
        "daily_summary": AgentIntent.DAILY_SUMMARY,
    }
    intent = intent_map.get(raw, AgentIntent.GENERAL)
    logger.info("Intent classified: %s (raw=%s)", intent, raw)
    return {**state, "intent": intent.value}


def _node_route(state: OrchestratorState) -> str:
    """Conditional edge: route to appropriate agent node."""
    return state.get("intent", AgentIntent.GENERAL.value)


def _node_knowledge(state: OrchestratorState) -> OrchestratorState:
    """Call the Knowledge Agent for nutrition lookup."""
    agent = KnowledgeAgent()
    extra = state.get("extra_data", {})
    portion_g = float(extra.get("portion_g", 100.0))

    nutrition = agent.get_nutrition(state["user_request"], portion_g)
    result = nutrition.model_dump()

    return {
        **state,
        "agent_result": result,
        "final_response": _format_nutrition_response(nutrition),
    }


def _node_meal_plan(state: OrchestratorState) -> OrchestratorState:
    """Call the Recommendation Agent for a meal plan."""
    agent = RecommendationAgent()
    profile = _deserialise_profile(state["profile"])
    plan = agent.generate_meal_plan(profile)

    return {
        **state,
        "agent_result": plan,
        "final_response": _format_plan_response(plan),
    }


def _node_health_advice(state: OrchestratorState) -> OrchestratorState:
    """Call the Health Advisory Agent."""
    agent = HealthAdvisoryAgent()
    profile = _deserialise_profile(state["profile"])
    advice = agent.get_health_advice(profile, state["user_request"])

    return {
        **state,
        "agent_result": {"advice": advice},
        "final_response": advice,
    }


def _node_log_meal(state: OrchestratorState) -> OrchestratorState:
    """Call the Food Log Agent for text meal logging."""
    agent = FoodLogAgent()
    profile = _deserialise_profile(state["profile"])
    extra = state.get("extra_data", {})
    meal_type = MealType(extra.get("meal_type", MealType.SNACK.value))
    daily_summary: DailySummary | None = None
    if ds_data := extra.get("daily_summary"):
        try:
            daily_summary = DailySummary.model_validate(ds_data)
        except Exception:
            pass

    log = agent.log_meal_text(
        profile=profile,
        description=state["user_request"],
        meal_type=meal_type,
        daily_summary=daily_summary,
    )

    return {
        **state,
        "agent_result": log.model_dump(mode="json"),
        "final_response": _format_meal_log_response(log),
    }


def _node_daily_summary(state: OrchestratorState) -> OrchestratorState:
    """Build a daily nutrition summary from provided meal logs."""
    agent = FoodLogAgent()
    profile = _deserialise_profile(state["profile"])
    extra = state.get("extra_data", {})

    raw_logs = extra.get("meal_logs", [])
    meal_logs: list[MealLog] = []
    for log_data in raw_logs:
        try:
            meal_logs.append(MealLog.model_validate(log_data))
        except Exception as exc:
            logger.warning("Skipping invalid meal log: %s", exc)

    summary = agent.build_daily_summary(profile, meal_logs)

    return {
        **state,
        "agent_result": summary.model_dump(mode="json"),
        "final_response": _format_daily_summary_response(summary),
    }


def _node_general(state: OrchestratorState) -> OrchestratorState:
    """Handle general nutrition questions directly with Granite LLM."""
    guardrails = GuardrailsFilter(use_llm_guardian=True)
    llm = get_llm_client()
    profile = _deserialise_profile(state["profile"])

    prompt = (
        f"<|system|>\nYou are an expert nutrition assistant. Answer the following "
        f"question for a {profile.age}-year-old {profile.sex} with "
        f"{profile.dietary_philosophy.value} dietary preferences. "
        f"Be specific, evidence-based, and concise.\n"
        f"<|user|>\n{state['user_request']}\n<|assistant|>"
    )
    response = guardrails.safe_generate(
        lambda: llm.generate(prompt),
        is_health_content=True,
    )
    return {
        **state,
        "agent_result": {"response": response},
        "final_response": response,
    }


def _node_error_handler(state: OrchestratorState) -> OrchestratorState:
    """Catch-all error handler node."""
    return {
        **state,
        "final_response": (
            "I encountered an error processing your request. "
            "Please try again or rephrase your question."
        ),
    }


# ─────────────────────────────────────────────────────────────
#  Graph construction
# ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    """Build and compile the LangGraph supervisor workflow."""
    graph = StateGraph(OrchestratorState)

    # Add nodes
    graph.add_node("classify", _node_classify)
    graph.add_node(AgentIntent.NUTRITION_LOOKUP.value, _node_knowledge)
    graph.add_node(AgentIntent.MEAL_PLAN.value, _node_meal_plan)
    graph.add_node(AgentIntent.HEALTH_ADVICE.value, _node_health_advice)
    graph.add_node(AgentIntent.LOG_MEAL.value, _node_log_meal)
    graph.add_node(AgentIntent.DAILY_SUMMARY.value, _node_daily_summary)
    graph.add_node(AgentIntent.GENERAL.value, _node_general)

    # Entry point
    graph.set_entry_point("classify")

    # Conditional routing after classification
    graph.add_conditional_edges(
        "classify",
        _node_route,
        {
            AgentIntent.NUTRITION_LOOKUP.value: AgentIntent.NUTRITION_LOOKUP.value,
            AgentIntent.MEAL_PLAN.value: AgentIntent.MEAL_PLAN.value,
            AgentIntent.HEALTH_ADVICE.value: AgentIntent.HEALTH_ADVICE.value,
            AgentIntent.LOG_MEAL.value: AgentIntent.LOG_MEAL.value,
            AgentIntent.DAILY_SUMMARY.value: AgentIntent.DAILY_SUMMARY.value,
            AgentIntent.GENERAL.value: AgentIntent.GENERAL.value,
        },
    )

    # All agents terminate at END
    for intent in AgentIntent:
        graph.add_edge(intent.value, END)

    return graph


# ─────────────────────────────────────────────────────────────
#  Orchestrator entry point
# ─────────────────────────────────────────────────────────────

class NutritionOrchestrator:
    """
    Main entry point for the multi-agent nutrition system.
    Wraps the compiled LangGraph workflow.
    """

    def __init__(self) -> None:
        self._graph = build_graph().compile()

    def run(
        self,
        user_request: str,
        profile: UserProfile,
        extra_data: dict | None = None,
    ) -> dict[str, Any]:
        """
        Execute the orchestrator for a user request.

        Args:
            user_request: Natural language request from the user.
            profile: UserProfile object.
            extra_data: Optional dict with auxiliary inputs
                        (portion_g, meal_type, meal_logs, audio_bytes, image_bytes, etc.)

        Returns:
            dict with keys: intent, agent_result, final_response
        """
        initial_state: OrchestratorState = {
            "user_request": user_request,
            "profile": profile.model_dump(mode="json"),
            "extra_data": extra_data or {},
            "intent": "",
            "agent_result": None,
            "final_response": "",
            "error": None,
        }

        try:
            final_state = self._graph.invoke(initial_state)
            return {
                "intent": final_state["intent"],
                "agent_result": final_state["agent_result"],
                "final_response": final_state["final_response"],
            }
        except Exception as exc:
            logger.error("Orchestrator error: %s", exc, exc_info=True)
            return {
                "intent": "error",
                "agent_result": None,
                "final_response": "An error occurred. Please try again.",
                "error": str(exc),
            }


# ─────────────────────────────────────────────────────────────
#  Response formatters
# ─────────────────────────────────────────────────────────────

def _deserialise_profile(profile_data: dict) -> UserProfile:
    return UserProfile.model_validate(profile_data)


def _format_nutrition_response(nutrition: Any) -> str:
    m = nutrition.macros
    return (
        f"**{nutrition.food_name}** ({nutrition.serving_size_g}g)\n\n"
        f"| Nutrient | Amount |\n|---|---|\n"
        f"| Calories | {m.calories:.1f} kcal |\n"
        f"| Protein | {m.protein_g:.1f}g |\n"
        f"| Carbohydrates | {m.carbohydrates_g:.1f}g |\n"
        f"| Fat | {m.fat_g:.1f}g |\n"
        f"| Fiber | {m.fiber_g:.1f}g |\n"
        f"| Sugar | {m.sugar_g:.1f}g |\n"
    )


def _format_plan_response(plan: dict) -> str:
    if "raw_plan" in plan:
        return plan["raw_plan"]
    return (
        f"**{plan.get('plan_name', '7-Day Meal Plan')}**\n\n"
        f"Daily Target: {plan.get('daily_calorie_target', 'N/A')} kcal\n\n"
        f"{plan.get('weekly_summary', '')}\n\n"
        "_Full plan available in the structured response._"
    )


def _format_meal_log_response(log: MealLog) -> str:
    items_str = "\n".join(
        f"  • {item.food_name}: {item.quantity_g}g ({item.nutrition.macros.calories:.0f} kcal)"
        if item.nutrition else f"  • {item.food_name}: {item.quantity_g}g"
        for item in log.items
    )
    alerts_str = "\n".join(f"  {a}" for a in log.alerts) if log.alerts else "  No alerts."
    return (
        f"**Meal Logged** ({log.meal_type.value.title()})\n\n"
        f"Items:\n{items_str}\n\n"
        f"Total: **{log.total_macros.calories:.0f} kcal** | "
        f"P: {log.total_macros.protein_g:.0f}g | "
        f"C: {log.total_macros.carbohydrates_g:.0f}g | "
        f"F: {log.total_macros.fat_g:.0f}g\n\n"
        f"Feedback:\n{alerts_str}"
    )


def _format_daily_summary_response(summary: DailySummary) -> str:
    pct = (summary.consumed.calories / summary.target.calories * 100) if summary.target.calories else 0
    feedback_str = "\n".join(f"  {f}" for f in summary.feedback) if summary.feedback else "  Keep it up!"
    return (
        f"**Daily Summary — {summary.date}**\n\n"
        f"Progress: **{summary.consumed.calories:.0f} / {summary.target.calories:.0f} kcal** ({pct:.0f}%)\n"
        f"Protein: {summary.consumed.protein_g:.0f}g | "
        f"Carbs: {summary.consumed.carbohydrates_g:.0f}g | "
        f"Fat: {summary.consumed.fat_g:.0f}g\n\n"
        f"Remaining: {summary.remaining.calories:.0f} kcal\n\n"
        f"Today's Insights:\n{feedback_str}"
    )
