"""
src/ui/app.py
─────────────────────────────────────────────────────────────
Streamlit Interactive Dashboard for the AI Nutrition Agent Platform.

Features:
  - User profile configuration panel
  - Daily Macro Progress Rings (Plotly donut charts)
  - Micronutrient Deficiency Radar chart
  - Multimodal Meal Logger (Text / Audio / Image)
  - 7-Day Meal Plan Generator with PDF export
  - Health Advisory interface

Run: streamlit run src/ui/app.py
"""
from __future__ import annotations

import base64
import io
import json
import queue
import uuid
from datetime import datetime
from pathlib import Path

import av
import numpy as np
import plotly.graph_objects as go
import requests
import streamlit as st
from plotly.subplots import make_subplots
from streamlit_webrtc import WebRtcMode, webrtc_streamer

# ─────────────────────────────────────────────────────────────
#  Configuration
# ─────────────────────────────────────────────────────────────

API_BASE = "http://localhost:8000/api/v1"
PAGE_TITLE = "🥗 AI Nutrition Agent"

st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon="🥗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
#  Custom CSS
# ─────────────────────────────────────────────────────────────

st.markdown(
    """
<style>
    .metric-card {
        background: #f7f8fa;
        border: 1px solid #e5e7eb;
        border-radius: 12px;
        padding: 16px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #3b82d4; }
    .metric-label { font-size: 0.85rem; color: #57606a; margin-top: 4px; }
    .alert-box {
        background: #fff8e1;
        border-left: 4px solid #f59e0b;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .success-box {
        background: #f0fdf4;
        border-left: 4px solid #22c55e;
        border-radius: 4px;
        padding: 10px 14px;
        margin: 6px 0;
    }
    .stTab [data-baseweb="tab"] { font-size: 0.95rem; }
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────────────────────
#  Session state initialisation
# ─────────────────────────────────────────────────────────────

# Audio frame queue — shared between WebRTC callback and main thread
_audio_queue: queue.Queue = queue.Queue()


def _init_session() -> None:
    defaults = {
        "user_id": str(uuid.uuid4()),
        "profile": None,
        "meal_logs": [],
        "daily_summary": None,
        "meal_plan": None,
        "recorded_audio_bytes": None,   # WAV bytes from live recorder
        "recording_done": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_session()

# ─────────────────────────────────────────────────────────────
#  API helpers
# ─────────────────────────────────────────────────────────────

def _api_post(endpoint: str, payload: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", json=payload, timeout=120)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API server. Start it with: `uvicorn src.api.main:app`")
    except requests.exceptions.HTTPError as e:
        st.error(f"API error: {e.response.text[:300]}")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


def _api_post_file(endpoint: str, files: dict, data: dict) -> dict | None:
    try:
        resp = requests.post(f"{API_BASE}{endpoint}", files=files, data=data, timeout=180)
        resp.raise_for_status()
        return resp.json()
    except requests.exceptions.ConnectionError:
        st.error("⚠️ Cannot connect to the API server.")
    except Exception as e:
        st.error(f"Request failed: {e}")
    return None


def _get_profile_dict() -> dict | None:
    return st.session_state.get("profile")


# ─────────────────────────────────────────────────────────────
#  Sidebar — User Profile
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/5/51/IBM_logo.svg", width=80)
    st.title("👤 My Profile")

    with st.form("profile_form"):
        name = st.text_input("Name", value="Alex", placeholder="Your name")
        col1, col2 = st.columns(2)
        with col1:
            age = st.number_input("Age", min_value=10, max_value=120, value=30)
            weight_kg = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.5)
        with col2:
            sex = st.selectbox("Sex", ["male", "female", "other"])
            height_cm = st.number_input("Height (cm)", min_value=100.0, max_value=250.0, value=170.0, step=0.5)

        activity = st.selectbox(
            "Activity Level",
            ["sedentary", "lightly_active", "moderately_active", "very_active", "super_active"],
            index=2,
        )
        diet_philosophy = st.selectbox(
            "Dietary Philosophy",
            ["omnivore", "vegetarian", "vegan", "pescatarian", "keto",
             "mediterranean", "halal", "kosher", "jain", "gluten_free"],
        )
        fitness_goal = st.selectbox(
            "Fitness Goal",
            ["fat_loss", "muscle_gain", "maintenance", "athletic_performance", "general_wellness"],
            index=2,
        )
        health_conditions = st.multiselect(
            "Health Conditions",
            ["type2_diabetes", "hypertension", "chronic_kidney_disease",
             "high_cholesterol", "irritable_bowel_syndrome", "celiac_disease"],
        )
        allergies_raw = st.text_input("Allergies (comma-separated)", placeholder="e.g., nuts, dairy")
        medications_raw = st.text_input("Medications (comma-separated)", placeholder="e.g., metformin")

        if st.form_submit_button("💾 Save Profile", use_container_width=True):
            st.session_state["profile"] = {
                "user_id": st.session_state["user_id"],
                "name": name,
                "age": age,
                "sex": sex,
                "weight_kg": weight_kg,
                "height_cm": height_cm,
                "activity_level": activity,
                "dietary_philosophy": diet_philosophy,
                "fitness_goal": fitness_goal,
                "health_conditions": health_conditions,
                "allergies": [a.strip() for a in allergies_raw.split(",") if a.strip()],
                "medications": [m.strip() for m in medications_raw.split(",") if m.strip()],
            }
            st.success("Profile saved!")
            st.rerun()

    if st.session_state["profile"]:
        p = st.session_state["profile"]
        # Quick BMR display
        if p["sex"] == "male":
            bmr = 10 * p["weight_kg"] + 6.25 * p["height_cm"] - 5 * p["age"] + 5
        else:
            bmr = 10 * p["weight_kg"] + 6.25 * p["height_cm"] - 5 * p["age"] - 161
        multipliers = {"sedentary": 1.2, "lightly_active": 1.375, "moderately_active": 1.55, "very_active": 1.725, "super_active": 1.9}
        tdee = bmr * multipliers.get(p["activity_level"], 1.55)
        adj = {"fat_loss": -500, "muscle_gain": 300, "maintenance": 0, "athletic_performance": 400, "general_wellness": 0}
        target = tdee + adj.get(p["fitness_goal"], 0)

        st.divider()
        st.metric("BMR", f"{bmr:.0f} kcal")
        st.metric("TDEE", f"{tdee:.0f} kcal")
        st.metric("Daily Target", f"{target:.0f} kcal")


# ─────────────────────────────────────────────────────────────
#  Main content — Tabs
# ─────────────────────────────────────────────────────────────

st.title(PAGE_TITLE)
st.caption("Powered by IBM Granite 3.0 · watsonx.ai · LangGraph")

# ── Demo mode banner ──────────────────────────────────────────
try:
    _hc = requests.get(f"{API_BASE.replace('/api/v1', '')}/health", timeout=3).json()
    if _hc.get("status") == "healthy":
        # Check if running in demo mode by probing a quick endpoint
        _dm_check = requests.post(
            f"{API_BASE}/nutrition-lookup",
            json={"food_name": "apple", "portion_g": 100},
            timeout=5,
        )
        if _dm_check.ok:
            _api_ok = True
        else:
            _api_ok = False
    else:
        _api_ok = False
except Exception:
    _api_ok = False

if not _api_ok:
    st.warning(
        "⚠️ **API server not reachable.** Start it with:\n"
        "```\nuvicorn src.api.main:app --host 0.0.0.0 --port 8000\n```",
        icon="⚠️",
    )
else:
    # Probe .env to show demo banner
    import os as _os
    _api_key = _os.environ.get("WATSONX_APIKEY", "")
    _placeholder_keys = {"your_ibm_cloud_api_key_here", "", "placeholder", "test-api-key"}
    if _api_key.strip() in _placeholder_keys:
        st.info(
            "🧪 **Demo Mode** — Running with mock AI responses. "
            "All features work end-to-end with realistic sample data. "
            "To activate real IBM Granite AI, add your credentials to `.env`.",
            icon="🧪",
        )

if not st.session_state["profile"]:
    st.info("👈 Configure your profile in the sidebar to get started.")
    st.stop()

profile = st.session_state["profile"]

tab_dashboard, tab_log, tab_plan, tab_health, tab_lookup = st.tabs([
    "📊 Dashboard",
    "🍽️ Log Meal",
    "📅 Meal Plan",
    "🩺 Health Advice",
    "🔍 Nutrition Lookup",
])

# ─────────────────────────────────────────────────────────────
#  TAB 1 — DASHBOARD
# ─────────────────────────────────────────────────────────────

with tab_dashboard:
    st.subheader(f"Today's Progress — {datetime.now().strftime('%A, %B %d')}")

    summary = st.session_state.get("daily_summary")

    if summary is None:
        st.info("No meals logged today yet. Use the **Log Meal** tab to get started.")
        consumed = {"calories": 0, "protein_g": 0, "carbohydrates_g": 0, "fat_g": 0, "fiber_g": 0}
        # Estimate targets from profile
        if profile["sex"] == "male":
            bmr = 10 * profile["weight_kg"] + 6.25 * profile["height_cm"] - 5 * profile["age"] + 5
        else:
            bmr = 10 * profile["weight_kg"] + 6.25 * profile["height_cm"] - 5 * profile["age"] - 161
        mults = {"sedentary": 1.2, "lightly_active": 1.375, "moderately_active": 1.55, "very_active": 1.725, "super_active": 1.9}
        tdee = bmr * mults.get(profile["activity_level"], 1.55)
        adj = {"fat_loss": -500, "muscle_gain": 300, "maintenance": 0, "athletic_performance": 400, "general_wellness": 0}
        target_cal = tdee + adj.get(profile["fitness_goal"], 0)
        target = {
            "calories": target_cal,
            "protein_g": (target_cal * 0.25) / 4,
            "carbohydrates_g": (target_cal * 0.45) / 4,
            "fat_g": (target_cal * 0.30) / 9,
            "fiber_g": 30,
        }
    else:
        consumed = summary.get("consumed", {})
        target = summary.get("target", {})

    # ── Macro Rings ───────────────────────────────────────────
    st.markdown("#### 🎯 Macro Progress Rings")

    def _donut(label: str, consumed_val: float, target_val: float, color: str) -> go.Figure:
        pct = min((consumed_val / target_val * 100) if target_val else 0, 100)
        fig = go.Figure(go.Pie(
            values=[pct, 100 - pct],
            hole=0.72,
            marker_colors=[color, "#f0f0f0"],
            showlegend=False,
            textinfo="none",
            hoverinfo="skip",
        ))
        fig.add_annotation(
            text=f"<b>{consumed_val:.0f}</b><br><span style='font-size:10px'>{label}</span>",
            x=0.5, y=0.5, showarrow=False, font_size=14, align="center",
        )
        fig.update_layout(
            height=180, margin=dict(t=10, b=10, l=10, r=10),
            paper_bgcolor="white", plot_bgcolor="white",
        )
        return fig

    cols = st.columns(5)
    rings = [
        ("Calories\n(kcal)", consumed.get("calories", 0), target.get("calories", 2000), "#3b82d4"),
        ("Protein\n(g)", consumed.get("protein_g", 0), target.get("protein_g", 120), "#22c55e"),
        ("Carbs\n(g)", consumed.get("carbohydrates_g", 0), target.get("carbohydrates_g", 250), "#f59e0b"),
        ("Fat\n(g)", consumed.get("fat_g", 0), target.get("fat_g", 65), "#ef4444"),
        ("Fiber\n(g)", consumed.get("fiber_g", 0), target.get("fiber_g", 30), "#8b5cf6"),
    ]
    for col, (label, val, tgt, color) in zip(cols, rings):
        with col:
            st.plotly_chart(_donut(label.split("\n")[0], val, tgt, color), use_container_width=True)
            remaining = max(0, tgt - val)
            st.caption(f"**{val:.0f}** / {tgt:.0f} · {remaining:.0f} remaining")

    # ── Micronutrient Radar ───────────────────────────────────
    st.markdown("#### 🧬 Micronutrient Status")

    micro_data = {}
    if summary:
        micro_data = summary.get("micronutrient_totals", {})

    # Daily reference values (% of RDA)
    rda_map = {
        "Vitamin A": (micro_data.get("vitamin_a_mcg", 0) / 900 * 100, "900 mcg"),
        "Vitamin C": (micro_data.get("vitamin_c_mg", 0) / 90 * 100, "90 mg"),
        "Vitamin D": (micro_data.get("vitamin_d_mcg", 0) / 15 * 100, "15 mcg"),
        "Vitamin B12": (micro_data.get("vitamin_b12_mcg", 0) / 2.4 * 100, "2.4 mcg"),
        "Calcium": (micro_data.get("calcium_mg", 0) / 1000 * 100, "1000 mg"),
        "Iron": (micro_data.get("iron_mg", 0) / 18 * 100, "18 mg"),
        "Magnesium": (micro_data.get("magnesium_mg", 0) / 420 * 100, "420 mg"),
        "Potassium": (micro_data.get("potassium_mg", 0) / 4700 * 100, "4700 mg"),
        "Zinc": (micro_data.get("zinc_mg", 0) / 11 * 100, "11 mg"),
    }
    categories = list(rda_map.keys())
    values = [min(v[0], 150) for v in rda_map.values()]  # cap at 150%

    fig_radar = go.Figure(go.Scatterpolar(
        r=values + [values[0]],
        theta=categories + [categories[0]],
        fill="toself",
        fillcolor="rgba(59, 130, 212, 0.2)",
        line=dict(color="#3b82d4", width=2),
        name="% of Daily RDA",
    ))
    fig_radar.add_shape(type="circle",
        xref="paper", yref="paper", x0=0.3, y0=0.3, x1=0.7, y1=0.7,
        line=dict(color="#22c55e", dash="dash"),
    )
    fig_radar.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 150], ticksuffix="%"),
        ),
        showlegend=False,
        height=380,
        margin=dict(t=20, b=20),
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    st.caption("Dashed circle = 100% of daily RDA. Values sourced from today's logged meals.")

    # ── Daily Feedback ────────────────────────────────────────
    if summary and summary.get("feedback"):
        st.markdown("#### 💡 Today's Insights")
        for fb in summary["feedback"]:
            st.markdown(f'<div class="success-box">{fb}</div>', unsafe_allow_html=True)

    # ── Refresh summary button ────────────────────────────────
    if st.button("🔄 Refresh Daily Summary"):
        with st.spinner("Calculating summary..."):
            result = _api_post("/daily-summary", {
                "profile": profile,
                "meal_logs": st.session_state["meal_logs"],
                "date": datetime.utcnow().date().isoformat(),
            })
            if result and result.get("status") == "success":
                st.session_state["daily_summary"] = result["daily_summary"]
                st.rerun()


# ─────────────────────────────────────────────────────────────
#  TAB 2 — LOG MEAL
# ─────────────────────────────────────────────────────────────

with tab_log:
    st.subheader("🍽️ Log Your Meal")

    meal_type_sel = st.selectbox(
        "Meal Type",
        ["breakfast", "lunch", "dinner", "snack"],
        key="log_meal_type",
    )

    input_method = st.radio(
        "Input Method",
        ["📝 Text", "🎤 Audio", "📷 Photo"],
        horizontal=True,
    )

    logged_result = None

    # ── Text input ────────────────────────────────────────────
    if input_method == "📝 Text":
        meal_text = st.text_area(
            "Describe your meal",
            placeholder="e.g. 2 scrambled eggs with whole wheat toast and a glass of orange juice",
            height=100,
        )
        if st.button("📊 Log & Analyse", use_container_width=True, key="log_text_btn"):
            if not meal_text.strip():
                st.warning("Please enter a meal description.")
            else:
                with st.spinner("Analysing your meal with IBM Granite..."):
                    result = _api_post("/log-meal", {
                        "profile": profile,
                        "description": meal_text,
                        "meal_type": meal_type_sel,
                        "daily_summary": st.session_state.get("daily_summary"),
                    })
                    if result and result.get("status") == "success":
                        logged_result = result["meal_log"]
                        st.session_state["meal_logs"].append(logged_result)
                        st.success("✅ Meal logged successfully!")

    # ── Live Audio Recorder ───────────────────────────────────
    elif input_method == "🎤 Audio":
        st.markdown("**🎙️ Record your meal description directly in the browser**")
        st.caption("Click **START** to begin recording, speak your meal, then click **STOP**.")

        # Collect PCM frames from WebRTC into module-level queue
        def _audio_frame_callback(frame: av.AudioFrame) -> av.AudioFrame:
            _audio_queue.put(frame.to_ndarray())
            return frame

        ctx = webrtc_streamer(
            key="meal-audio-recorder",
            mode=WebRtcMode.SENDONLY,
            audio_frame_callback=_audio_frame_callback,
            media_stream_constraints={"audio": True, "video": False},
            rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
            translations={
                "start": "▶ START Recording",
                "stop": "⏹ STOP Recording",
                "select_device": "Select Microphone",
            },
        )

        # Drain queue while recording is active
        if ctx.state.playing:
            st.info("🔴 Recording… speak your meal description clearly.")
            frames: list[np.ndarray] = []
            try:
                while True:
                    frames.append(_audio_queue.get_nowait())
            except queue.Empty:
                pass
            if frames:
                # Accumulate in session state
                existing = st.session_state.get("_audio_frames", [])
                st.session_state["_audio_frames"] = existing + frames

        # When recording stops, encode to WAV and store
        if not ctx.state.playing and st.session_state.get("_audio_frames"):
            import struct
            import wave

            frames = st.session_state["_audio_frames"]
            pcm = np.concatenate(frames, axis=-1).flatten().astype(np.int16)

            wav_buf = io.BytesIO()
            with wave.open(wav_buf, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)        # 16-bit PCM
                wf.setframerate(48000)    # WebRTC default sample rate
                wf.writeframes(pcm.tobytes())
            wav_bytes = wav_buf.getvalue()

            st.session_state["recorded_audio_bytes"] = wav_bytes
            st.session_state["_audio_frames"] = []   # clear accumulator
            st.success(f"✅ Recording captured — {len(wav_bytes) / 1024:.1f} KB")

        # Show playback + log button once recording is captured
        audio_bytes = st.session_state.get("recorded_audio_bytes")
        if audio_bytes:
            st.audio(audio_bytes, format="audio/wav")
            col_log, col_clear = st.columns([3, 1])
            with col_clear:
                if st.button("🗑️ Discard", use_container_width=True):
                    st.session_state["recorded_audio_bytes"] = None
                    st.rerun()
            with col_log:
                if st.button("🎙️ Transcribe & Log", use_container_width=True, type="primary"):
                    with st.spinner("Transcribing & analysing with IBM Granite..."):
                        result = _api_post_file(
                            "/log-meal/audio",
                            files={"audio_file": ("recording.wav", audio_bytes, "audio/wav")},
                            data={
                                "user_id": profile["user_id"],
                                "name": profile["name"],
                                "age": profile["age"],
                                "sex": profile["sex"],
                                "weight_kg": profile["weight_kg"],
                                "height_cm": profile["height_cm"],
                                "meal_type": meal_type_sel,
                            },
                        )
                        if result and result.get("status") == "success":
                            logged_result = result["meal_log"]
                            st.session_state["meal_logs"].append(logged_result)
                            st.session_state["recorded_audio_bytes"] = None
                            if result.get("transcript"):
                                st.info(f"📝 Transcript: *{result['transcript']}*")
                            st.success("✅ Audio meal logged!")

    # ── Image input ───────────────────────────────────────────
    elif input_method == "📷 Photo":
        image_file = st.file_uploader(
            "Upload plate/meal photo",
            type=["jpg", "jpeg", "png", "webp"],
            key="image_uploader",
        )
        if image_file:
            st.image(image_file, caption="Your meal photo", width=300)
        if image_file and st.button("🔭 Analyse Photo & Log", use_container_width=True):
            with st.spinner("Analysing with Granite Vision..."):
                result = _api_post_file(
                    "/analyze-image",
                    files={"image_file": (image_file.name, image_file.getvalue(), image_file.type)},
                    data={
                        "user_id": profile["user_id"],
                        "name": profile["name"],
                        "age": profile["age"],
                        "sex": profile["sex"],
                        "weight_kg": profile["weight_kg"],
                        "height_cm": profile["height_cm"],
                        "meal_type": meal_type_sel,
                    },
                )
                if result and result.get("status") == "success":
                    logged_result = result["meal_log"]
                    st.session_state["meal_logs"].append(logged_result)
                    if result.get("vision_analysis"):
                        st.info(f"👁️ Vision analysis: *{result['vision_analysis'][:200]}*")
                    st.success("✅ Photo meal logged!")

    # ── Display logged result ─────────────────────────────────
    if logged_result:
        st.divider()
        st.subheader("📋 Meal Analysis")

        items = logged_result.get("items", [])
        if items:
            col_a, col_b = st.columns([2, 1])
            with col_a:
                st.markdown("**Food Items Detected:**")
                table_rows = []
                for item in items:
                    nut = item.get("nutrition", {})
                    macros = nut.get("macros", {}) if nut else {}
                    table_rows.append({
                        "Food": item["food_name"],
                        "Amount (g)": item["quantity_g"],
                        "Calories": f"{macros.get('calories', 0):.0f}",
                        "Protein (g)": f"{macros.get('protein_g', 0):.1f}",
                        "Carbs (g)": f"{macros.get('carbohydrates_g', 0):.1f}",
                        "Fat (g)": f"{macros.get('fat_g', 0):.1f}",
                    })
                st.dataframe(table_rows, use_container_width=True)

            with col_b:
                totals = logged_result.get("total_macros", {})
                st.markdown("**Meal Totals:**")
                st.markdown(f'<div class="metric-card"><div class="metric-value">{totals.get("calories", 0):.0f}</div><div class="metric-label">kcal</div></div>', unsafe_allow_html=True)
                st.write("")
                cols3 = st.columns(3)
                for col3, (label, key) in zip(cols3, [("Protein", "protein_g"), ("Carbs", "carbohydrates_g"), ("Fat", "fat_g")]):
                    with col3:
                        st.metric(label, f"{totals.get(key, 0):.0f}g")

        alerts = logged_result.get("alerts", [])
        if alerts:
            st.markdown("**💡 Feedback:**")
            for alert in alerts:
                st.markdown(f'<div class="alert-box">{alert}</div>', unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
#  TAB 3 — MEAL PLAN
# ─────────────────────────────────────────────────────────────

with tab_plan:
    st.subheader("📅 7-Day Personalised Meal Plan")
    st.markdown(
        f"Generate a customised meal plan for **{profile.get('name', 'you')}** "
        f"based on your {profile.get('dietary_philosophy', 'omnivore')} diet "
        f"and {profile.get('fitness_goal', 'maintenance')} goal."
    )

    if st.button("🤖 Generate My Meal Plan", use_container_width=True, type="primary"):
        with st.spinner("IBM Granite is crafting your personalised 7-day plan..."):
            result = _api_post("/recommend-plan", {"profile": profile})
            if result and result.get("status") == "success":
                st.session_state["meal_plan"] = result
                st.success("✅ Your meal plan is ready!")

    plan_data = st.session_state.get("meal_plan")
    if plan_data:
        meal_plan = plan_data.get("meal_plan", {})
        macro_summary = plan_data.get("macro_summary", {})

        # Macro summary banner
        if macro_summary:
            m_cols = st.columns(4)
            for col, (label, key) in zip(m_cols, [
                ("Daily Calories", "calories"), ("Protein", "protein_g"),
                ("Carbs", "carbohydrates_g"), ("Fat", "fat_g")
            ]):
                unit = "kcal" if key == "calories" else "g"
                col.metric(label, f"{macro_summary.get(key, 0):.0f} {unit}")

        st.divider()

        # Plan name and summary
        if meal_plan.get("plan_name"):
            st.markdown(f"### {meal_plan['plan_name']}")
        if meal_plan.get("weekly_summary"):
            st.info(meal_plan["weekly_summary"])

        # Day-by-day display
        days = meal_plan.get("days", [])
        if days:
            day_tabs = st.tabs([f"Day {d.get('day', i+1)}: {d.get('day_name', '')}" for i, d in enumerate(days)])
            for day_tab, day in zip(day_tabs, days):
                with day_tab:
                    st.caption(f"**Total estimated calories: {day.get('total_calories', 'N/A')} kcal**")
                    for meal in day.get("meals", []):
                        with st.expander(f"🍽️ {meal.get('meal_type', '').title()} — {meal.get('meal_name', '')} (~{meal.get('estimated_calories', '?')} kcal)"):
                            for item in meal.get("items", []):
                                st.markdown(f"• **{item.get('food_name')}** — {item.get('quantity_g', '?')}g — *{item.get('description', '')}*")
                            if meal.get("notes"):
                                st.info(f"💡 {meal['notes']}")
        elif meal_plan.get("raw_plan"):
            st.markdown(meal_plan["raw_plan"])

        # Grocery list
        grocery = meal_plan.get("grocery_list", {})
        if grocery:
            with st.expander("🛒 Weekly Grocery List"):
                for category, items in grocery.items():
                    st.markdown(f"**{category.title()}**")
                    for item in items:
                        st.markdown(f"  - {item}")

        # Substitutions
        subs = meal_plan.get("substitution_options", [])
        if subs:
            with st.expander("🔄 Substitution Options"):
                for sub in subs:
                    st.markdown(f"• {sub}")

        # PDF export
        st.divider()
        if st.button("📄 Export Plan as PDF"):
            try:
                from fpdf import FPDF  # type: ignore

                pdf = FPDF()
                pdf.add_page()
                pdf.set_font("Helvetica", "B", 16)
                pdf.cell(0, 10, meal_plan.get("plan_name", "7-Day Meal Plan"), ln=True)
                pdf.set_font("Helvetica", "", 11)
                pdf.cell(0, 8, f"Generated for: {profile.get('name', '')}", ln=True)
                pdf.cell(0, 8, f"Daily Target: {meal_plan.get('daily_calorie_target', '?')} kcal", ln=True)
                pdf.ln(4)
                for day in days:
                    pdf.set_font("Helvetica", "B", 12)
                    pdf.cell(0, 9, f"Day {day.get('day')}: {day.get('day_name', '')}", ln=True)
                    pdf.set_font("Helvetica", "", 10)
                    for meal in day.get("meals", []):
                        pdf.cell(0, 7, f"  {meal.get('meal_type', '').title()}: {meal.get('meal_name', '')}", ln=True)
                        for item in meal.get("items", []):
                            pdf.cell(0, 6, f"    - {item.get('food_name')}: {item.get('quantity_g')}g", ln=True)
                    pdf.ln(2)

                pdf_bytes = pdf.output(dest="S").encode("latin-1")
                b64 = base64.b64encode(pdf_bytes).decode()
                st.markdown(
                    f'<a href="data:application/pdf;base64,{b64}" download="meal_plan.pdf">'
                    "⬇️ Download PDF</a>",
                    unsafe_allow_html=True,
                )
            except ImportError:
                st.warning("fpdf2 not installed. Run: `pip install fpdf2`")


# ─────────────────────────────────────────────────────────────
#  TAB 4 — HEALTH ADVICE
# ─────────────────────────────────────────────────────────────

with tab_health:
    st.subheader("🩺 Preventive Health & Nutrition Advice")
    st.caption(
        "Powered by Granite 3 + Granite Guardian safety filters. "
        "All advice includes clinical disclaimers."
    )

    if profile.get("health_conditions"):
        st.info(f"Your conditions: **{', '.join(profile['health_conditions'])}**")
    else:
        st.info("No health conditions set. Add them in your profile for personalised clinical guidance.")

    health_question = st.text_area(
        "Your health nutrition question",
        value="What specific foods should I eat and avoid for my conditions?",
        height=100,
        key="health_q",
    )

    if st.button("🤖 Get Health Advice", use_container_width=True, type="primary", key="health_btn"):
        with st.spinner("Consulting clinical guidelines with IBM Granite..."):
            result = _api_post("/health-advice", {
                "profile": profile,
                "question": health_question,
            })
            if result and result.get("status") == "success":
                st.markdown("### 📋 Clinical Nutrition Advice")
                st.markdown(result["advice"])

    # Drug interaction checker
    st.divider()
    st.subheader("💊 Drug-Nutrient Interaction Checker")
    if profile.get("medications"):
        st.write(f"Your medications: **{', '.join(profile['medications'])}**")
        food_check = st.text_input(
            "Foods recently eaten (comma-separated)",
            placeholder="e.g. grapefruit, spinach, milk",
        )
        if st.button("Check Interactions"):
            with st.spinner("Checking interactions..."):
                result = _api_post("/chat", {
                    "profile": profile,
                    "message": f"Check drug-nutrient interactions for my medications with these foods: {food_check}",
                    "extra_data": {},
                })
                if result:
                    st.markdown(result.get("final_response", ""))
    else:
        st.info("Add your medications in the profile sidebar to use the interaction checker.")


# ─────────────────────────────────────────────────────────────
#  TAB 5 — NUTRITION LOOKUP
# ─────────────────────────────────────────────────────────────

with tab_lookup:
    st.subheader("🔍 Food Nutrition Lookup")
    st.caption("Query the USDA FoodData Central database via RAG-powered semantic search.")

    col_food, col_portion = st.columns([3, 1])
    with col_food:
        food_query = st.text_input("Food name", placeholder="e.g. brown rice cooked, raw broccoli, grilled salmon")
    with col_portion:
        portion = st.number_input("Portion (g)", min_value=1.0, max_value=2000.0, value=100.0, step=10.0)

    if st.button("🔍 Look Up", use_container_width=True):
        if not food_query.strip():
            st.warning("Enter a food name.")
        else:
            with st.spinner(f"Searching nutrition data for '{food_query}'..."):
                result = _api_post("/nutrition-lookup", {
                    "food_name": food_query,
                    "portion_g": portion,
                })
                if result and result.get("status") == "success":
                    nut = result["nutrition"]
                    macros = nut.get("macros", {})
                    micros = nut.get("micros", {})

                    st.markdown(f"### {nut.get('food_name', food_query)} ({portion}g)")

                    # Macro bar chart
                    fig_bar = go.Figure(go.Bar(
                        x=["Protein", "Carbs", "Fat", "Fiber", "Sugar"],
                        y=[
                            macros.get("protein_g", 0),
                            macros.get("carbohydrates_g", 0),
                            macros.get("fat_g", 0),
                            macros.get("fiber_g", 0),
                            macros.get("sugar_g", 0),
                        ],
                        marker_color=["#22c55e", "#f59e0b", "#ef4444", "#8b5cf6", "#f97316"],
                        text=[
                            f"{macros.get('protein_g', 0):.1f}g",
                            f"{macros.get('carbohydrates_g', 0):.1f}g",
                            f"{macros.get('fat_g', 0):.1f}g",
                            f"{macros.get('fiber_g', 0):.1f}g",
                            f"{macros.get('sugar_g', 0):.1f}g",
                        ],
                        textposition="outside",
                    ))
                    fig_bar.update_layout(
                        height=280,
                        margin=dict(t=20, b=20),
                        yaxis_title="grams",
                        showlegend=False,
                    )

                    col_macro, col_info = st.columns([2, 1])
                    with col_macro:
                        st.plotly_chart(fig_bar, use_container_width=True)
                    with col_info:
                        st.metric("Calories", f"{macros.get('calories', 0):.0f} kcal")
                        st.metric("Glycemic Index", nut.get("glycemic_index") or "N/A")
                        allergens = nut.get("allergens", [])
                        if allergens:
                            st.warning(f"⚠️ Allergens: {', '.join(allergens)}")
                        else:
                            st.success("✅ No common allergens")

                    # Micronutrient table
                    with st.expander("🧬 Full Micronutrient Profile"):
                        micro_rows = [
                            {"Nutrient": "Vitamin A", "Amount": f"{micros.get('vitamin_a_mcg', 0):.1f} mcg", "RDA": "900 mcg"},
                            {"Nutrient": "Vitamin C", "Amount": f"{micros.get('vitamin_c_mg', 0):.1f} mg", "RDA": "90 mg"},
                            {"Nutrient": "Vitamin D", "Amount": f"{micros.get('vitamin_d_mcg', 0):.1f} mcg", "RDA": "15 mcg"},
                            {"Nutrient": "Vitamin B12", "Amount": f"{micros.get('vitamin_b12_mcg', 0):.2f} mcg", "RDA": "2.4 mcg"},
                            {"Nutrient": "Folate", "Amount": f"{micros.get('folate_mcg', 0):.0f} mcg", "RDA": "400 mcg"},
                            {"Nutrient": "Calcium", "Amount": f"{micros.get('calcium_mg', 0):.0f} mg", "RDA": "1000 mg"},
                            {"Nutrient": "Iron", "Amount": f"{micros.get('iron_mg', 0):.2f} mg", "RDA": "18 mg"},
                            {"Nutrient": "Magnesium", "Amount": f"{micros.get('magnesium_mg', 0):.0f} mg", "RDA": "420 mg"},
                            {"Nutrient": "Potassium", "Amount": f"{micros.get('potassium_mg', 0):.0f} mg", "RDA": "4700 mg"},
                            {"Nutrient": "Sodium", "Amount": f"{micros.get('sodium_mg', 0):.0f} mg", "RDA": "< 2300 mg"},
                            {"Nutrient": "Zinc", "Amount": f"{micros.get('zinc_mg', 0):.2f} mg", "RDA": "11 mg"},
                            {"Nutrient": "Selenium", "Amount": f"{micros.get('selenium_mcg', 0):.1f} mcg", "RDA": "55 mcg"},
                        ]
                        st.dataframe(micro_rows, use_container_width=True, hide_index=True)
