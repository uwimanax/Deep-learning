# ---- Avoid pyarrow DLL issues on Windows ----
import os
os.environ["PANDAS_NO_PYARROW"] = "1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping


# =========================================================
# 0. Page config + Custom CSS
# =========================================================
st.set_page_config(
    page_title="Smart Day Planner",
    page_icon="🗓️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
.main {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
    max-width: 1300px;
}

/* ---------- Hero header ---------- */
.hero {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
    border-radius: 20px;
    padding: 2rem 2.5rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 10px 40px rgba(99, 102, 241, 0.35);
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.hero-icon {
    background: rgba(255,255,255,0.15);
    border-radius: 16px;
    padding: 14px;
    display: flex;
    align-items: center;
    justify-content: center;
    backdrop-filter: blur(10px);
}
.hero h1 {
    color: #ffffff;
    font-size: 2.1rem;
    font-weight: 800;
    margin: 0;
    letter-spacing: -0.5px;
}
.hero p {
    color: rgba(255,255,255,0.9);
    font-size: 1rem;
    margin: 0.25rem 0 0 0;
}

/* ---------- Section titles ---------- */
.section-title {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    color: #e2e8f0;
    font-size: 1.25rem;
    font-weight: 700;
    margin: 1.75rem 0 1rem 0;
    padding-bottom: 0.5rem;
    border-bottom: 2px solid rgba(99,102,241,0.3);
}

/* ---------- KPI cards ---------- */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
    gap: 1rem;
    margin: 1rem 0;
}
.kpi-card {
    background: linear-gradient(145deg, #1e293b, #0f172a);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 1.25rem;
    box-shadow: 0 4px 20px rgba(0,0,0,0.25);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 8px 30px rgba(99,102,241,0.35);
}
.kpi-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    color: #94a3b8;
    font-size: 0.85rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
.kpi-value {
    color: #f1f5f9;
    font-size: 1.9rem;
    font-weight: 800;
    margin-top: 0.5rem;
    letter-spacing: -0.5px;
}

/* ---------- Feedback banner ---------- */
.feedback {
    border-radius: 14px;
    padding: 1.1rem 1.4rem;
    margin: 1rem 0;
    display: flex;
    align-items: center;
    gap: 0.9rem;
    font-weight: 600;
    font-size: 1rem;
}
.feedback-good {
    background: linear-gradient(90deg, rgba(16,185,129,0.15), rgba(16,185,129,0.05));
    border-left: 5px solid #10b981;
    color: #6ee7b7;
}
.feedback-ok {
    background: linear-gradient(90deg, rgba(59,130,246,0.15), rgba(59,130,246,0.05));
    border-left: 5px solid #3b82f6;
    color: #93c5fd;
}
.feedback-warn {
    background: linear-gradient(90deg, rgba(245,158,11,0.15), rgba(245,158,11,0.05));
    border-left: 5px solid #f59e0b;
    color: #fcd34d;
}

/* ---------- Recommendation box ---------- */
.recommend {
    background: linear-gradient(145deg, #1e1b4b, #172554);
    border: 1px solid rgba(139,92,246,0.4);
    border-radius: 16px;
    padding: 1.5rem;
    color: #e0e7ff;
    display: flex;
    align-items: center;
    gap: 1rem;
    font-size: 1.02rem;
    line-height: 1.6;
}

/* ---------- Progress bar ---------- */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, #6366f1, #ec4899);
}

/* ---------- Sidebar ---------- */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
    border-right: 1px solid rgba(99,102,241,0.2);
}
section[data-testid="stSidebar"] h2 {
    color: #e2e8f0;
    font-size: 1.1rem;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border: none;
    border-radius: 12px;
    padding: 0.75rem 1.5rem;
    font-weight: 700;
    font-size: 1rem;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(99,102,241,0.4);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(139,92,246,0.55);
    background: linear-gradient(90deg, #8b5cf6, #ec4899);
}
</style>
""", unsafe_allow_html=True)


# =========================================================
# 1. Inline SVG icon helper
# =========================================================
def icon(name, size=20, color="#c7d2fe"):
    """Return inline SVG markup for a given icon name."""
    paths = {
        "moon":      '<path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>',
        "list":      '<line x1="8" y1="6" x2="21" y2="6"/><line x1="8" y1="12" x2="21" y2="12"/><line x1="8" y1="18" x2="21" y2="18"/><line x1="3" y1="6" x2="3.01" y2="6"/><line x1="3" y1="12" x2="3.01" y2="12"/><line x1="3" y1="18" x2="3.01" y2="18"/>',
        "clock":     '<circle cx="12" cy="12" r="10"/><polyline points="12 6 12 12 16 14"/>',
        "coffee":    '<path d="M18 8h1a4 4 0 0 1 0 8h-1"/><path d="M2 8h16v9a4 4 0 0 1-4 4H6a4 4 0 0 1-4-4V8z"/><line x1="6" y1="1" x2="6" y2="4"/><line x1="10" y1="1" x2="10" y2="4"/><line x1="14" y1="1" x2="14" y2="4"/>',
        "users":     '<path d="M17 21v-2a4 4 0 0 0-4-4H5a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M23 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/>',
        "zap":       '<polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/>',
        "alert":     '<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>',
        "sunrise":   '<path d="M17 18a5 5 0 0 0-10 0"/><line x1="12" y1="2" x2="12" y2="9"/><line x1="4.22" y1="10.22" x2="5.64" y2="11.64"/><line x1="1" y1="18" x2="3" y2="18"/><line x1="21" y1="18" x2="23" y2="18"/><line x1="18.36" y1="11.64" x2="19.78" y2="10.22"/><line x1="23" y1="22" x2="1" y2="22"/><polyline points="8 6 12 2 16 6"/>',
        "target":    '<circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/>',
        "check":     '<polyline points="20 6 9 17 4 12"/>',
        "trend":     '<polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/>',
        "bulb":      '<path d="M9 18h6"/><path d="M10 22h4"/><path d="M15.09 14c.18-.98.65-1.74 1.41-2.5A4.65 4.65 0 0 0 18 8 6 6 0 0 0 6 8c0 1 .23 2.23 1.5 3.5A4.61 4.61 0 0 1 8.91 14"/>',
        "brain":     '<path d="M9.5 2A2.5 2.5 0 0 1 12 4.5v15a2.5 2.5 0 0 1-4.96.44 2.5 2.5 0 0 1-2.96-3.08 3 3 0 0 1-.34-5.58 2.5 2.5 0 0 1 1.32-4.24 2.5 2.5 0 0 1 1.98-3A2.5 2.5 0 0 1 9.5 2z"/><path d="M14.5 2A2.5 2.5 0 0 0 12 4.5v15a2.5 2.5 0 0 0 4.96.44 2.5 2.5 0 0 0 2.96-3.08 3 3 0 0 0 .34-5.58 2.5 2.5 0 0 0-1.32-4.24 2.5 2.5 0 0 0-1.98-3A2.5 2.5 0 0 0 14.5 2z"/>',
        "chart":     '<line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/>',
        "calendar":  '<rect x="3" y="4" width="18" height="18" rx="2" ry="2"/><line x1="16" y1="2" x2="16" y2="6"/><line x1="8" y1="2" x2="8" y2="6"/><line x1="3" y1="10" x2="21" y2="10"/>',
    }
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{size}" height="{size}" '
        f'viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="2" '
        f'stroke-linecap="round" stroke-linejoin="round">{paths.get(name, "")}</svg>'
    )


# =========================================================
# 2. Data generation
# =========================================================
@st.cache_data
def generate_data(n=2000, seed=42):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({
        "sleep_hours":   rng.uniform(4, 9, n),
        "tasks_planned": rng.integers(1, 15, n),
        "task_minutes":  rng.integers(60, 600, n),
        "break_minutes": rng.integers(0, 120, n),
        "meetings":      rng.integers(0, 8, n),
        "energy":        rng.integers(1, 11, n),
        "stress":        rng.integers(1, 11, n),
        "start_hour":    rng.integers(5, 12, n),
    })
    score = (
        0.15 * df["sleep_hours"]
        + 0.04 * df["energy"]
        - 0.05 * df["stress"]
        - 0.02 * df["meetings"]
        - 0.0005 * df["task_minutes"]
        + 0.01 * df["break_minutes"]
        - 0.01 * df["tasks_planned"]
        + rng.normal(0, 0.08, n)
    )
    score = (score - score.min()) / (score.max() - score.min())
    df["completed_ratio"] = score
    return df


# =========================================================
# 3. Model training
# =========================================================
@st.cache_resource(show_spinner="Training deep learning model...")
def train_model(df):
    TARGET = "completed_ratio"
    X = df.drop(columns=[TARGET]).astype(float)
    y = df[TARGET].astype(float)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    model = Sequential([
        Dense(64, activation="relu", input_shape=(X_train_s.shape[1],)),
        Dropout(0.2),
        Dense(32, activation="relu"),
        Dropout(0.2),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid"),
    ])
    model.compile(optimizer="adam", loss="mse", metrics=["mae"])
    model.fit(
        X_train_s, y_train,
        validation_split=0.2,
        epochs=60,
        batch_size=32,
        callbacks=[EarlyStopping(monitor="val_loss", patience=8,
                                 restore_best_weights=True)],
        verbose=0,
    )
    mae = float(model.evaluate(X_test_s, y_test, verbose=0)[1])
    return model, scaler, X.columns.tolist(), mae


# =========================================================
# 4. Train
# =========================================================
df = generate_data()
model, scaler, features, mae = train_model(df)


# =========================================================
# 5. Hero header
# =========================================================
st.markdown(f"""
<div class="hero">
    <div class="hero-icon">{icon("calendar", 42, "#ffffff")}</div>
    <div>
        <h1>Smart Day Planner</h1>
        <p>Deep Learning Powered Productivity Prediction · Model MAE: {mae:.3f}</p>
    </div>
</div>
""", unsafe_allow_html=True)


# =========================================================
# 6. Sidebar inputs
# =========================================================
st.sidebar.markdown(f"""
<div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
    {icon("target", 22, "#a5b4fc")}
    <span style="color:#e2e8f0;font-size:1.1rem;font-weight:700;">Your Day Plan</span>
</div>
""", unsafe_allow_html=True)

u = {}
u["sleep_hours"]   = st.sidebar.slider("Sleep (hours)", 4.0, 9.0, 7.0, 0.5)
u["tasks_planned"] = st.sidebar.slider("Tasks planned", 1, 15, 6)
u["task_minutes"]  = st.sidebar.slider("Total task minutes", 60, 600, 240, 10)
u["break_minutes"] = st.sidebar.slider("Break minutes", 0, 120, 30, 5)
u["meetings"]      = st.sidebar.slider("Meetings today", 0, 8, 2)
u["energy"]        = st.sidebar.slider("Energy level (1-10)", 1, 10, 7)
u["stress"]        = st.sidebar.slider("Stress level (1-10)", 1, 10, 4)
u["start_hour"]    = st.sidebar.slider("Start hour (24h)", 5, 12, 8)


# =========================================================
# 7. Predict
# =========================================================
if st.button("Predict My Productivity", use_container_width=True):

    row = pd.DataFrame([u])[features]
    scaled = scaler.transform(row)
    ratio = float(model.predict(scaled, verbose=0)[0][0])
    ratio = max(0.0, min(1.0, ratio))

    st.markdown(
        f'<div class="section-title">{icon("chart", 22)}Prediction Results</div>',
        unsafe_allow_html=True,
    )

    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">{icon("target", 16)}Completion</div>
            <div class="kpi-value">{ratio*100:.1f}%</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("check", 16)}Tasks Finished</div>
            <div class="kpi-value">{round(ratio * u['tasks_planned'])} / {u['tasks_planned']}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("clock", 16)}Focus Time</div>
            <div class="kpi-value">{int(ratio * u['task_minutes'])} min</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">{icon("alert", 16)}Remaining</div>
            <div class="kpi-value">{int((1-ratio) * u['task_minutes'])} min</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.progress(ratio)

    if ratio >= 0.75:
        st.markdown(f"""
        <div class="feedback feedback-good">
            {icon("check", 22, "#6ee7b7")}
            Great plan! You are likely to complete most of it.
        </div>
        """, unsafe_allow_html=True)
    elif ratio >= 0.5:
        st.markdown(f"""
        <div class="feedback feedback-ok">
            {icon("zap", 22, "#93c5fd")}
            Decent plan. Maybe trim one task to be safer.
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="feedback feedback-warn">
            {icon("alert", 22, "#fcd34d")}
            Plan looks overloaded. Try fewer tasks or longer breaks.
        </div>
        """, unsafe_allow_html=True)

    st.markdown(
        f'<div class="section-title">{icon("trend", 22)}What-If Analysis</div>',
        unsafe_allow_html=True,
    )
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Completion vs. Tasks Planned**")
        sweep = np.arange(1, 16)
        preds = [
            float(model.predict(scaler.transform(
                pd.DataFrame([{**u, "tasks_planned": int(t)}])[features]
            ), verbose=0)[0][0])
            for t in sweep
        ]
        st.line_chart(
            pd.DataFrame({"Tasks Planned": sweep, "Completion": preds})
              .set_index("Tasks Planned")
        )

    with col2:
        st.markdown("**Completion vs. Sleep Hours**")
        sweep = np.linspace(4, 9, 20)
        preds = [
            float(model.predict(scaler.transform(
                pd.DataFrame([{**u, "sleep_hours": float(s)}])[features]
            ), verbose=0)[0][0])
            for s in sweep
        ]
        st.line_chart(
            pd.DataFrame({"Sleep Hours": sweep, "Completion": preds})
              .set_index("Sleep Hours")
        )

    st.markdown(
        f'<div class="section-title">{icon("bulb", 22)}Smart Recommendation</div>',
        unsafe_allow_html=True,
    )

    best_ratio, best_plan = ratio, u.copy()
    for t in range(u["tasks_planned"], max(1, u["tasks_planned"] - 5), -1):
        r = {**u, "tasks_planned": t}
        p = float(model.predict(scaler.transform(
            pd.DataFrame([r])[features]
        ), verbose=0)[0][0])
        if p > best_ratio:
            best_ratio, best_plan = p, r

    if best_ratio > ratio + 0.03:
        msg = (
            f"Try planning <b>{best_plan['tasks_planned']} tasks</b> instead of "
            f"{u['tasks_planned']} — predicted completion rises to "
            f"<b>{best_ratio*100:.1f}%</b>."
        )
    else:
        msg = "Your current plan is already near-optimal. Stick with it!"

    st.markdown(f"""
    <div class="recommend">
        {icon("brain", 28, "#a5b4fc")}
        <div>{msg}</div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("View raw model output"):
        st.json({
            "predicted_completed_ratio": round(ratio, 4),
            "input_vector": u,
        })

st.caption("Built with TensorFlow/Keras + Streamlit · Multi-Layer Perceptron regression")
