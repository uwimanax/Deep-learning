# Smart Day Planner

A deep-learning-powered productivity predictor with a styled Streamlit dashboard.

Enter your day plan (sleep, tasks, breaks, meetings, energy, stress), and a
neural network predicts what percentage of your tasks you'll realistically
complete — plus a smart recommendation for a better plan.

---

## Demo

- **Hero banner** with gradient + custom SVG icons
- **Sidebar sliders** for 8 daily planning inputs
- **KPI cards** showing completion %, tasks finished, focus time, remaining time
- **Progress bar** with gradient fill
- **Color-coded feedback** (great / decent / overloaded)
- **What-If line charts** (tasks vs completion, sleep vs completion)
- **Smart recommendation card** suggesting an improved plan
- **Raw output expander** for debugging

---

## Project Structure

```
DayPlannerAI/
├── app.py              # Everything (data, model, UI)
├── requirements.txt    # Dependencies
└── README.md           # You are here
```

Single-file app. No `train.py`, no `data.csv`, no extra scripts.

---

## How It Works

### 1. Synthetic Data (generated on first run)
2,000 simulated day plans with 8 features:

| Feature | Range | Meaning |
|---|---|---|
| `sleep_hours` | 4.0 – 9.0 | Hours slept |
| `tasks_planned` | 1 – 15 | Number of tasks |
| `task_minutes` | 60 – 600 | Total task duration |
| `break_minutes` | 0 – 120 | Total break time |
| `meetings` | 0 – 8 | Meetings scheduled |
| `energy` | 1 – 10 | Subjective energy level |
| `stress` | 1 – 10 | Subjective stress level |
| `start_hour` | 5 – 12 | Start hour of day |

Target: `completed_ratio` (0.0 – 1.0) — normalized productivity score.

### 2. Deep Learning Model
A Multi-Layer Perceptron (MLP) regression model built in Keras:

```
Input (8 features)
      ↓
Dense(64) + ReLU → Dropout(0.2)
      ↓
Dense(32) + ReLU → Dropout(0.2)
      ↓
Dense(16) + ReLU
      ↓
Dense(1) + Sigmoid   →   output: 0.0 – 1.0
```

- **Loss:** Mean Squared Error
- **Optimizer:** Adam
- **Callbacks:** EarlyStopping (patience=8)
- **Epochs:** 60 max
- **Batch size:** 32

### 3. Streamlit UI
- `@st.cache_data` — generates data once
- `@st.cache_resource` — trains model once per session
- Custom CSS for a modern dark theme
- Inline SVG icons (no external icon library)
- Interactive sliders + real-time prediction

---

## Installation

### 1. Clone or create the folder
```bash
mkdir DayPlannerAI
cd DayPlannerAI
```

### 2. Create a virtual environment
**Windows:**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the app
```bash
streamlit run app.py
```

The app opens automatically at `http://localhost:8501`.

On first run, the model trains (about 20–40 seconds). Subsequent runs are instant
thanks to Streamlit's caching.

---

## Usage

1. Move the **sidebar sliders** to match your planned day.
2. Click **Predict My Productivity**.
3. View:
   - KPI cards (completion %, tasks finished, focus time, remaining time)
   - Progress bar
   - Feedback banner
   - Two What-If charts
   - Recommendation card
4. Adjust sliders and predict again to compare scenarios.

---

## Troubleshooting

### `ImportError: DLL load failed while importing _compute`
Windows Smart App Control is blocking `pyarrow`.

**Fix options:**
1. Disable **Smart App Control** (Windows Security → App & browser control)
2. The app already sets `PANDAS_NO_PYARROW=1` to avoid loading `pyarrow` at all.
3. Ensure you don't have `pyarrow` in `requirements.txt`.

### `DLL load failed while importing _pywrap_tensorflow_internal`
Missing Microsoft Visual C++ Redistributable.

**Fix:** Download and install the x64 version:
https://aka.ms/vs/17/release/vc_redist.x64.exe

### Model training is slow
First run takes 20–40 seconds. After that, it's cached.
To make it faster, reduce `epochs=60` to `epochs=20` in `app.py`.

### App doesn't open in browser
Manually visit: `http://localhost:8501`

---

## Tech Stack

| Layer | Tool |
|---|---|
| UI | Streamlit |
| Styling | Custom CSS injected via `st.markdown` |
| Icons | Inline SVG (no library) |
| ML Framework | TensorFlow / Keras |
| Data Processing | pandas, NumPy |
| Preprocessing | scikit-learn (StandardScaler) |
| Caching | Streamlit `@st.cache_data` / `@st.cache_resource` |

---

## Customization

### Change model size
In `app.py`, edit the `Sequential([...])` block:
```python
Dense(128, activation="relu"),   # try bigger layers
Dropout(0.3),
Dense(64, activation="relu"),
```

### Change theme colors
Edit the CSS block at the top of `app.py`:
```css
.hero {
    background: linear-gradient(120deg, #6366f1 0%, #8b5cf6 50%, #ec4899 100%);
}
```

### Change input features
Edit `generate_data()` and the sidebar slider section.

---

## License

Free to use for learning, demos, and portfolio projects.

---

## Acknowledgements

Built with TensorFlow, Streamlit, and scikit-learn.
