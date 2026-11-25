# AI-Powered Multi-Agent Nutrition Platform
### IBM University Engagement — Project Submission

| Field | Details |
|---|---|
| **Student Name** | Shagun Harkawat |
| **Email** | 25etccs162@technonjr.org |
| **Mobile (WhatsApp)** | 7976984694 |
| **Domain** | Healthcare / Nutrition & Wellness |
| **Project Title** | AI-Powered Multi-Agent Nutrition Platform |
| **IBM Certificate** | Getting Started with Artificial Intelligence — 24 Nov 2026 |
| **Platform** | IBM Bob · IBM watsonx.ai · IBM Granite 3.0 |

---

## 📌 Problem Statement

Good nutrition is the foundation of health, yet individuals consistently struggle with diet planning, portion control, and tracking nutritional intake. Information about healthy diets, disease-specific nutrition, and food composition is scattered across unreliable sources.

**Without personalised, real-time guidance, most people fail to:**
- Maintain balanced diets suited to their health conditions
- Track daily macro and micronutrient intake
- Understand drug-nutrient interactions with their medications
- Get culturally aware, goal-driven meal recommendations

---

## 💡 Proposed Solution

An **AI-Powered Multi-Agent Nutrition Platform** built on IBM Bob, IBM watsonx.ai, and IBM Granite 3.0 models. The system uses a Retrieval-Augmented Generation (RAG) pipeline over USDA FoodData Central and clinical dietary guidelines, orchestrated by a LangGraph supervisor that routes user requests to four specialist agents.

---

## 🏗️ Architecture

```
User Request (Text / Audio / Image)
            │
            ▼
┌───────────────────────────────────────────────────┐
│        LangGraph Orchestrator / Supervisor         │
│   Intent Classification → Routes to Sub-Agent     │
└──────┬──────────┬────────────┬────────────┬───────┘
       │          │            │            │
       ▼          ▼            ▼            ▼
  Knowledge  Recommendation  Health     Food Log
   Agent       Agent         Agent       Agent
 (RAG/USDA) (Meal Plans)  (Clinical)  (Multimodal)
       │          │            │            │
       └──────────┴────────────┴────────────┘
                        │
              IBM Granite 3-8b-instruct
                 ibm/granite-3-8b-instruct
              IBM Granite Vision (image analysis)
              IBM Granite Guardian (safety checks)
              IBM Granite Embedding (RAG vectors)
                   ChromaDB Vector Store
```

---

## 🤖 Multi-Agent System

### 1. Knowledge Agent (`knowledge_agent.py`)
- Implements **RAG over USDA FoodData Central** indexed in ChromaDB
- Uses **Granite Embedding** to encode 10 seed foods + extensible to full USDA DB
- Returns exact calories, macros (protein, carbs, fat, fiber), and 15 micronutrients per food
- Scales nutrition data to any portion weight

### 2. Recommendation Agent (`recommendation_agent.py`)
- Accepts full user profile: age, sex, weight, height, activity level, dietary philosophy (Vegan, Keto, Halal, Jain, Mediterranean, etc.), allergies, and fitness goal
- Calculates **BMR (Mifflin-St Jeor) and TDEE** automatically
- Generates customisable **7-day meal plans** with portion sizes, grocery lists, and ingredient substitution options
- Calorie target adjusted per goal: −500 kcal (fat loss), +300 kcal (muscle gain)

### 3. Health Advisory Agent (`health_agent.py`)
- Specialises in **preventive health nutrition** for chronic conditions
- Covers Type 2 Diabetes (low-GI/GL), Hypertension (DASH), CKD (renal diet), High Cholesterol (TLC)
- Checks **drug-nutrient interactions** (e.g. grapefruit + statins, Metformin + B12)
- All outputs protected by **Granite Guardian** with automatic clinical disclaimers

### 4. Food Log & Feedback Agent (`food_log_agent.py`)
- **Text input**: parses natural language meal descriptions
- **Audio input**: in-browser WebRTC recorder → Whisper STT / Watson STT transcription
- **Image input**: plate photo → Granite Vision identifies food items and estimates portion weights
- Deducts from daily macro budget and generates real-time feedback alerts

---

## 🛠️ Technology Stack

| Layer | Technology |
|---|---|
| **LLM / Foundation Model** | `ibm/granite-3-8b-instruct` via ibm-watsonx-ai SDK |
| **Vision** | `ibm/granite-vision-3-2-2b` for meal photo analysis |
| **Safety** | `ibm/granite-guardian-3-8b` for content moderation |
| **Embeddings** | `ibm/granite-embedding-278m-multilingual` |
| **Agent Orchestration** | LangGraph (supervisor graph with conditional routing) |
| **RAG / Vector DB** | ChromaDB with Granite Embedding functions |
| **Backend API** | FastAPI + Uvicorn (Python 3.11) |
| **Data Validation** | Pydantic v2 |
| **Frontend Dashboard** | Streamlit + Plotly |
| **Speech-to-Text** | IBM Watson STT (cloud) / OpenAI Whisper (local fallback) |
| **Audio Capture** | streamlit-webrtc (in-browser live recording) |
| **Deployment** | Docker + IBM Cloud Code Engine ready |

---

## 📁 Project Structure

```
nutrition-agent/
├── docker/
│   ├── Dockerfile                    # Python 3.11-slim + ffmpeg + healthcheck
│   └── docker-compose.yml            # API + UI services with volume mounts
├── data/
│   ├── sample_usda_foods.json        # 10 USDA foods with full macros & micros
│   └── disease_guidelines.json       # Clinical guidelines: T2DM, HTN, CKD, Cholesterol
├── src/
│   ├── config.py                     # pydantic-settings — all env vars + BMR/TDEE helpers
│   ├── models/
│   │   ├── nutrition.py              # NutritionData, Macronutrients, Micronutrients
│   │   ├── user.py                   # UserProfile, ActivityLevel, FitnessGoal enums
│   │   └── meal.py                   # MealLog, MealItem, DailySummary, MacroBudget
│   ├── rag/
│   │   ├── ingest.py                 # ChromaDB indexing with Granite Embeddings
│   │   └── retriever.py              # FoodRetriever, GuidelineRetriever
│   ├── agents/
│   │   ├── orchestrator.py           # LangGraph supervisor — 6 routing nodes
│   │   ├── knowledge_agent.py        # RAG-powered food nutrition lookup
│   │   ├── recommendation_agent.py   # 7-day personalised meal plan generation
│   │   ├── health_agent.py           # Clinical health advisor + drug interaction checks
│   │   ├── food_log_agent.py         # Text / Audio / Image meal logger + feedback
│   │   └── guardrails.py             # Granite Guardian 2-stage safety filter
│   ├── services/
│   │   ├── watsonx_service.py        # IBM Granite clients (LLM, Vision, Embedding, Guardian)
│   │   │                             # Includes Demo Mode with MockLLMClient
│   │   └── stt_service.py            # Watson STT + Whisper fallback
│   ├── api/
│   │   └── main.py                   # FastAPI — 9 REST endpoints
│   └── ui/
│       └── app.py                    # Streamlit dashboard — 5 tabs, macro rings, radar chart
├── tests/
│   ├── conftest.py                   # IBM SDK stub + env injection (offline testing)
│   ├── test_agents.py                # 24 fully-mocked agent unit tests
│   └── test_rag.py                   # 13 RAG pipeline tests
├── requirements.txt
├── .env.example
├── pyproject.toml
└── README.md
```

---

## ⚡ Quick Start

### Prerequisites
- Python 3.11+
- IBM Cloud account with watsonx.ai access (or run in **Demo Mode** without credentials)

### 1. Clone & Install

```bash
git clone https://github.com/shagun-harkawat/nutrition-agent-ibm
cd nutrition-agent
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure Credentials

```bash
cp .env.example .env
```

Edit `.env`:
```
WATSONX_APIKEY=your_ibm_cloud_api_key
WATSONX_URL=https://us-south.ml.cloud.ibm.com
WATSONX_PROJECT_ID=your_project_id
```

> **Demo Mode:** If credentials are left as placeholders, the app automatically uses `MockLLMClient` — all features work end-to-end with realistic sample data. No IBM account required to explore the UI.

### 3. Index the Knowledge Base

```bash
python -m src.rag.ingest
# Output: ✅ Ingestion complete: 10 foods, 4 guidelines
```

### 4. Start the API

```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Interactive docs: http://localhost:8000/docs
```

### 5. Launch the Dashboard

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
| `POST` | `/api/v1/log-meal/audio` | Log meal via audio recording (Whisper / Watson STT) |
| `POST` | `/api/v1/analyze-image` | Analyse plate photo (Granite Vision) |
| `POST` | `/api/v1/health-advice` | Condition-specific dietary advice + drug-nutrient checks |
| `POST` | `/api/v1/nutrition-lookup` | RAG-powered food nutrition lookup |
| `POST` | `/api/v1/daily-summary` | Daily macro progress, micronutrient totals, AI feedback |
| `POST` | `/api/v1/chat` | Natural language interface (full orchestrator) |

### Sample — Log a Meal

```bash
curl -X POST http://localhost:8000/api/v1/log-meal \
  -H "Content-Type: application/json" \
  -d '{
    "profile": {
      "user_id": "user-001",
      "name": "Shagun",
      "age": 20,
      "sex": "female",
      "weight_kg": 55,
      "height_cm": 162,
      "activity_level": "lightly_active",
      "fitness_goal": "maintenance"
    },
    "description": "toasted bread, banana smoothie, dry fruits",
    "meal_type": "breakfast"
  }'
```

**Response:**
```json
{
  "status": "success",
  "meal_log": {
    "items": [
      { "food_name": "toasted bread", "quantity_g": 60, "nutrition": { "macros": { "calories": 90 } } },
      { "food_name": "banana smoothie", "quantity_g": 120, "nutrition": { "macros": { "calories": 180 } } },
      { "food_name": "dry fruits", "quantity_g": 30, "nutrition": { "macros": { "calories": 45 } } }
    ],
    "total_macros": { "calories": 315, "protein_g": 16.8, "carbohydrates_g": 46.0, "fat_g": 10.5 }
  }
}
```

---

## 📊 Streamlit Dashboard — 5 Tabs

| Tab | Features |
|---|---|
| **📊 Dashboard** | Plotly donut rings for Calories / Protein / Carbs / Fat / Fiber · 9-axis micronutrient radar vs % of daily RDA · AI daily insights |
| **🍽️ Log Meal** | Text input · **Live in-browser audio recorder** (WebRTC) · Plate photo upload (Granite Vision) · Per-item macro table · Feedback alerts |
| **📅 Meal Plan** | 7-day plan generator · Day-by-day expandable meals · Grocery list · Substitutions · PDF export |
| **🩺 Health Advice** | Condition-specific clinical advice · Drug-nutrient interaction checker · Granite Guardian safety filter |
| **🔍 Nutrition Lookup** | Macro bar chart · Glycemic Index · Allergen badges · Full micronutrient table vs RDA |

---

## 🔒 Safety & Guardrails

The **Granite Guardian** layer (`src/agents/guardrails.py`) applies at two stages:

1. **Input screening** — Regex pre-screen + `ibm/granite-guardian-3-8b` blocks harmful medical requests
2. **Output screening** — All health responses evaluated before delivery
3. **Clinical disclaimers** — Automatically appended to every health advisory response

---

## 🧬 Supported Health Conditions

| Condition | Dietary Protocol | Key Numeric Targets |
|---|---|---|
| **Type 2 Diabetes** | Low-GI/GL diet | GI < 55, GL < 10 per meal, 25–38g fiber/day |
| **Hypertension** | DASH diet | Sodium < 1500mg, 4700mg potassium/day |
| **CKD (Stages 3–5)** | Renal diet | Protein 0.6–0.8g/kg, Potassium < 2000mg/day |
| **High Cholesterol** | TLC diet | Saturated fat < 7%, 10–25g soluble fiber/day |

---

## 🧪 Running Tests

```bash
# All 37 tests — fully mocked, no IBM credentials needed
pytest tests/ -v

# With coverage report
pytest tests/ -v --cov=src --cov-report=html

# Individual suites
pytest tests/test_agents.py -v   # 24 agent tests
pytest tests/test_rag.py -v      # 13 RAG tests
```

**Test results: 37/37 passed ✅**

---

## 🐳 Docker

```bash
# Build and run everything
cd docker
docker compose up --build

# API:       http://localhost:8000
# Dashboard: http://localhost:8501
```

---

## 📦 Key Dependencies

| Package | Version | Purpose |
|---|---|---|
| `ibm-watsonx-ai` | ≥1.1.2 | IBM Granite LLM, Vision, Embedding, Guardian |
| `langgraph` | ≥0.2.0 | Multi-agent supervisor graph |
| `langchain` | ≥0.3.0 | LLM abstractions |
| `chromadb` | ≥0.5.0 | Vector database for RAG |
| `fastapi` | ≥0.115.0 | REST API |
| `streamlit` | ≥1.38.0 | Interactive dashboard |
| `streamlit-webrtc` | latest | In-browser live audio recording |
| `plotly` | ≥5.24.0 | Macro rings and radar charts |
| `pydantic` | ≥2.8.0 | Data validation |
| `openai-whisper` | latest | Local speech-to-text fallback |
| `fpdf2` | ≥2.7.9 | PDF meal plan export |

---

## 🔮 Future Scope

1. **Wearable Integration** — Connect with smartwatches / fitness trackers for real-time calorie and activity data, enabling fully dynamic dietary recommendations
2. **Multilingual Voice Assistant** — Extend STT and LLM to support Hindi, Tamil, and other regional languages for wider accessibility
3. **Full USDA Database** — Index all 300,000+ USDA FoodData Central entries for complete food coverage
4. **Personalised Learning** — Reinforcement learning from user feedback to improve meal plan personalisation over time
5. **EHR Integration** — Connect with hospital electronic health records for clinical-grade dietary management

---

## 📄 License

MIT License

---

> ⚕️ **Disclaimer:** This application is for informational and general wellness purposes only. It does not constitute professional medical advice, diagnosis, or treatment. Always consult a qualified healthcare professional before making dietary or medical decisions.
