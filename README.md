# AI Nutrition Agent — IBM Granite 3.0 · watsonx.ai · LangGraph

<div align="center">

```
╔══════════════════════════════════════════════════════╗
║          AI-Powered Multi-Agent Nutrition Platform    ║
║   IBM Granite 3.0 · watsonx.ai · LangGraph · FastAPI ║
╚══════════════════════════════════════════════════════╝
```

</div>

A production-grade, AI-powered nutrition platform built on **IBM Granite 3.0**, **LangGraph multi-agent orchestration**, and **RAG over USDA FoodData Central**. It delivers personalised dietary insights, multimodal meal tracking, preventive health advice, and an interactive Streamlit dashboard.

---

## 🏗️ Architecture

```
User Request
     │
     ▼
┌─────────────────────────────────────────────────────────┐
│          LangGraph Orchestrator / Supervisor             │
│  Classifies intent → routes to specialist sub-agent     │
└──────┬───────────┬──────────────┬──────────────┬────────┘
       │           │              │              │
       ▼           ▼              ▼              ▼
  Knowledge    Recommendation  Health        Food Log
   Agent         Agent         Agent          Agent
  (RAG/USDA)  (Meal Plans)  (Clinical/     (Text/Audio/
                            Drug-Check)      Image)
       │           │              │              │
       └───────────┴──────────────┴──────────────┘
                           │
                    IBM Granite 3.0
                   (ibm-watsonx-ai SDK)
                     Granite Guardian
                      (Safety Filter)
                     ChromaDB + RAG
                (Granite Embedding Model)
```

### Sub-Agents

| Agent | Responsibility | Key Technology |
|---|---|---|
| **Knowledge Agent** | Calorie/macro/micronutrient lookup | RAG → ChromaDB → Granite 3 |
| **Recommendation Agent** | 7-day personalised meal plans | BMR/TDEE calc, Granite 3 |
| **Health Advisory Agent** | Chronic condition nutrition, drug-nutrient checks | Clinical guidelines RAG, Granite Guardian |
| **Food Log Agent** | Multimodal meal ingestion + feedback | Granite Vision, Whisper STT, Granite 3 |

---

## 📁 Project Structure

```
nutrition-agent/
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── data/
│   ├── sample_usda_foods.json        # USDA seed database (10 foods)
│   └── disease_guidelines.json       # Clinical dietary guidelines (4 conditions)
├── src/
│   ├── config.py                     # Settings via pydantic-settings
│   ├── models/
│   │   ├── nutrition.py              # NutritionData, Macronutrients, Micronutrients
│   │   ├── user.py                   # UserProfile, BMR/TDEE calculations
│   │   └── meal.py                   # MealLog, DailySummary
│   ├── rag/
│   │   ├── ingest.py                 # ChromaDB indexing with Granite Embeddings
│   │   └── retriever.py              # FoodRetriever, GuidelineRetriever
│   ├── agents/
│   │   ├── orchestrator.py           # LangGraph supervisor graph
│   │   ├── knowledge_agent.py        # RAG-powered nutrition lookup
│   │   ├── recommendation_agent.py   # Meal plan generation
│   │   ├── health_agent.py           # Clinical health advisor
│   │   ├── food_log_agent.py         # Multimodal food logger
│   │   └── guardrails.py             # Granite Guardian safety filter
│   ├── services/
│   │   ├── watsonx_service.py        # IBM Granite LLM, Vision, Embedding, Guardian
│   │   └── stt_service.py            # Watson STT / Whisper fallback
│   ├── api/
│   │   └── main.py                   # FastAPI REST endpoints
│   └── ui/
│       └── app.py                    # Streamlit dashboard
├── tests/
│   ├── test_agents.py                # Agent unit tests (fully mocked)
│   └── test_rag.py                   # RAG pipeline tests
├── requirements.txt
├── .env.example
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- IBM Cloud account with watsonx.ai access
- A watsonx.ai project with Granite model access

### 1. Clone & Install

```bash
git clone <repo-url>
cd nutrition-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
# Edit .env and fill in your IBM credentials:
#   WATSONX_APIKEY=...
#   WATSONX_URL=https://us-south.ml.cloud.ibm.com
#   WATSONX_PROJECT_ID=...
```

### 3. Index the Knowledge Base (RAG)

```bash
python -m src.rag.ingest
# Output: ✅ Ingestion complete: 10 foods, 4 guidelines
```

### 4. Start the API Server

```bash
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000
# API docs: http://localhost:8000/docs
```

### 5. Launch the Streamlit Dashboard

```bash
streamlit run src/ui/app.py
# Dashboard: http://localhost:8501
```

---

## 🌐 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Service health check |
| `POST` | `/api/v1/recommend-plan` | Generate 7-day personalised meal plan |
| `POST` | `/api/v1/log-meal` | Log meal via text description |
| `POST` | `/api/v1/log-meal/audio` | Log meal via audio recording |
| `POST` | `/api/v1/analyze-image` | Log meal via plate photo (Granite Vision) |
| `POST` | `/api/v1/health-advice` | Get condition-specific dietary advice |
| `POST` | `/api/v1/nutrition-lookup` | Look up nutrition data for a food |
| `POST` | `/api/v1/daily-summary` | Get daily macro progress & AI feedback |
| `POST` | `/api/v1/chat` | Natural language interface (full orchestrator) |

### Example Request: Log a Meal

```bash
curl -X POST http://localhost:8000/api/v1/log-meal \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "user_id": "user-001",
      "name": "Alex",
      "age": 30,
      "sex": "male",
      "weight_kg": 75,
      "height_cm": 175,
      "activity_level": "moderately_active",
      "fitness_goal": "muscle_gain"
    },
    "description": "2 scrambled eggs with whole wheat toast and orange juice",
    "meal_type": "breakfast"
  }'
```

### Example Request: Get Meal Plan

```bash
curl -X POST http://localhost:8000/api/v1/recommend-plan \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "user_id": "user-002",
      "name": "Sam",
      "age": 45,
      "sex": "female",
      "weight_kg": 68,
      "height_cm": 165,
      "dietary_philosophy": "mediterranean",
      "health_conditions": ["hypertension"],
      "fitness_goal": "general_wellness"
    }
  }'
```

---

## 🐳 Docker Deployment

### Local Docker Compose

```bash
cd docker
docker compose up --build
# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

### IBM Cloud Code Engine

```bash
# Build and push image
ibmcloud cr build --dockerfile docker/Dockerfile --tag us.icr.io/<namespace>/nutrition-agent:latest .

# Deploy API
ibmcloud ce application create \
  --name nutrition-agent-api \
  --image us.icr.io/<namespace>/nutrition-agent:latest \
  --port 8000 \
  --env-from-secret nutrition-agent-secrets \
  --min-scale 1

# Deploy Dashboard
ibmcloud ce application create \
  --name nutrition-agent-ui \
  --image us.icr.io/<namespace>/nutrition-agent:latest \
  --port 8501 \
  --command streamlit \
  --argument "run,src/ui/app.py,--server.port=8501,--server.address=0.0.0.0"
```

---

## 🧪 Running Tests

```bash
# All tests (mocked — no API key required)
pytest tests/ -v

# With coverage
pytest tests/ -v --cov=src --cov-report=html

# Individual test files
pytest tests/test_agents.py -v
pytest tests/test_rag.py -v
```

Tests use `unittest.mock` to patch all IBM watsonx.ai and ChromaDB calls, ensuring they run entirely offline.

---

## 🔒 Safety & Guardrails

The **Granite Guardian** safety layer (`src/agents/guardrails.py`) applies at two stages:

1. **Input screening** — Regex + LLM pattern matching blocks dangerous medical requests
2. **Output screening** — All health-related outputs are checked by `ibm/granite-guardian-3-8b`
3. **Clinical disclaimers** — Automatically appended to all health advisory responses

---

## 🧬 Supported Health Conditions

| Condition | Dietary Protocol | Key Constraints |
|---|---|---|
| Type 2 Diabetes | Low-GI/GL | GI < 55, GL < 10, 25-38g fiber/day |
| Hypertension | DASH Diet | Sodium < 1500mg, 4700mg potassium/day |
| CKD (Stages 3-5) | Renal Diet | Low K/P/Na, 0.6-0.8g protein/kg |
| High Cholesterol | TLC Diet | Sat fat < 7%, 10-25g soluble fiber |

---

## 📦 Key Dependencies

| Package | Purpose |
|---|---|
| `ibm-watsonx-ai` | IBM Granite LLM, Vision, Embedding, Guardian |
| `langgraph` | Multi-agent supervisor graph |
| `langchain` | LLM abstractions |
| `chromadb` | Vector database for RAG |
| `fastapi` | REST API framework |
| `streamlit` | Interactive dashboard |
| `plotly` | Macro rings and radar charts |
| `pydantic v2` | Data validation |
| `openai-whisper` | Local speech-to-text fallback |
| `fpdf2` | PDF meal plan export |

---

## 📄 License

MIT License — See LICENSE file.

---

> **Disclaimer:** This application is for informational and wellness purposes only. It is not a substitute for professional medical advice. Always consult a qualified healthcare provider for medical decisions.
