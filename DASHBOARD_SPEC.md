# Customer Retention Analytics — Churn Intelligence Dashboard
## Agent Specification & Build Prompt

---

## READ THIS FIRST — WHAT YOU ARE BUILDING

You are building a professional, portfolio-grade Streamlit dashboard for a freelance
churn analyst. This is NOT a generic data science demo. It is a client-facing piece
that must look like a consultancy product — polished, intentional, and visually
confident.

The analyst presenting this dashboard is a retention analytics specialist targeting
subscription and service-based businesses. The dashboard demonstrates analytical
capability to potential clients. Every design and copy decision should reinforce
that this person thinks in business outcomes, not just model metrics.

**App title:** Customer Retention Analytics — Churn Intelligence Dashboard

**Do not start building until you have read this entire file.**

---

## PROJECT FILES

All files live in the same folder as `app.py`. Do not change file paths.

| File | Purpose |
|---|---|
| `processed_churn.csv` | Primary display data — 7,032 rows, human-readable values |
| `featured_churn.csv` | Model input data — 7,032 rows, scaled and encoded |
| `logistic_regression.pkl` | Trained sklearn LogisticRegression — load with joblib |
| `app.py` | The Streamlit app you are building |

**Do NOT use** `WA_Fn-UseC_-Telco-Customer-Churn.csv` — it has 7,043 rows due to
preprocessing drops and will cause index misalignment errors.

---

## DATA LOADING — FOLLOW THIS EXACTLY

```python
import pandas as pd
import numpy as np
import joblib

@st.cache_data
def load_data():
    df = pd.read_csv('processed_churn.csv')          # readable values
    df_feat = pd.read_csv('featured_churn.csv')       # model input
    model = joblib.load('logistic_regression.pkl')

    X = df_feat.drop('Churn', axis=1)                # 31 feature columns
    probs = model.predict_proba(X)[:, 1]             # churn probabilities

    df['ChurnProbability'] = probs
    df['RiskTier'] = pd.cut(
        probs,
        bins=[0, 0.40, 0.70, 1.0],
        labels=['Low', 'Medium', 'High']
    )
    return df, model
```

**Critical facts about the data:**
- `processed_churn.csv` and `featured_churn.csv` are row-aligned (both 7,032 rows).
  No merging needed — index positions correspond directly.
- `Churn` column in processed_churn.csv is integer (0 = retained, 1 = churned).
  It is NOT 'Yes'/'No'.
- `MonthlyCharges` in processed_churn.csv are real dollar values (e.g. 29.85, 56.95).
  Use this file for all display values. Never display scaled values from featured_churn.
- `Contract` values: 'Month-to-month', 'One year', 'Two year'
- `PaymentMethod` values: 'Electronic check', 'Mailed check',
  'Bank transfer (automatic)', 'Credit card (automatic)'

---

## CONFIRMED METRICS — USE THESE EXACT NUMBERS

These have been pre-verified. Compute them from data at runtime but they should
match these values. If they don't, something is wrong with data loading.

| Metric | Value |
|---|---|
| Total customers | 7,032 |
| Churned customers | 1,869 |
| Overall churn rate | 26.6% |
| Monthly revenue at risk | $139,130.85 |
| Top 20% targeting churn rate | 67.9% |
| Targeting efficiency | 2.55× vs baseline |

### Risk Tier Distribution
| Tier | Probability | Count |
|---|---|---|
| High | > 0.70 | ~1,800 |
| Medium | 0.40 – 0.70 | ~1,736 |
| Low | < 0.40 | ~3,496 |

### Churn by Contract Type
| Contract | Churn Rate |
|---|---|
| Month-to-month | 42.7% |
| One year | 11.3% |
| Two year | 2.8% |

### Churn by Payment Method
| Payment Method | Churn Rate |
|---|---|
| Electronic check | 45.3% |
| Mailed check | 19.2% |
| Bank transfer (automatic) | 16.7% |
| Credit card (automatic) | 15.3% |

---

## VISUAL DESIGN — FOLLOW PRECISELY

This dashboard has two modes: Light (default) and Dark. Both use the same accent
colors. The theme toggle lives in the sidebar.

### Color System

```python
LIGHT = {
    "bg":           "#F5F0E8",   # warm off-white, creamy — like Claude.ai
    "card_bg":      "#FFFFFF",
    "card_shadow":  "0 2px 12px rgba(0,0,0,0.07)",
    "text":         "#1C1917",
    "text_muted":   "#78716C",
    "border":       "#E7E5E4",
    "sidebar_bg":   "#EDE8DF",
}

DARK = {
    "bg":           "#1A1F2E",   # deep charcoal — not pure black
    "card_bg":      "#242938",
    "card_shadow":  "0 2px 12px rgba(0,0,0,0.3)",
    "text":         "#F5F5F4",
    "text_muted":   "#A8A29E",
    "border":       "#374151",
    "sidebar_bg":   "#151929",
}

ACCENT = {
    "teal":         "#0D9488",   # primary accent — deep teal
    "teal_hover":   "#0F766E",
    "high":         "#EF4444",   # risk colors
    "medium":       "#F59E0B",
    "low":          "#10B981",
    "high_bg":      "#FEF2F2",   # soft background tints for risk badges
    "medium_bg":    "#FFFBEB",
    "low_bg":       "#F0FDF4",
}
```

### Typography

Load via st.markdown at app start:

```python
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}
h1, h2, h3 {
    font-family: 'DM Serif Display', serif;
}
</style>
""", unsafe_allow_html=True)
```

### Cards

Every metric card must follow this structure:
- Rounded corners: 12px border-radius
- Left border: 4px solid teal accent
- Inner padding: 20px 24px
- Subtle shadow from color system above
- Metric value in large bold DM Serif Display
- Label in small DM Sans muted color

### Charts (ALL charts must follow these rules)

- Library: **Plotly Express / Plotly Graph Objects only**. No matplotlib. No seaborn.
- Transparent backgrounds: always set `plot_bgcolor='rgba(0,0,0,0)'`
  and `paper_bgcolor='rgba(0,0,0,0)'`
- Hide Plotly toolbar: `st.plotly_chart(fig, use_container_width=True,
  config={'displayModeBar': False})`
- No gridlines on bar charts
- Font family on all charts: 'DM Sans'
- Font color adapts to theme (dark text on light, light text on dark)
- Risk tier charts always use exact risk colors: High=#EF4444, Medium=#F59E0B, Low=#10B981
- All other charts use teal (#0D9488) as primary color
- Add a subtle horizontal reference line where meaningful (e.g., average churn rate line
  on bar charts)

### What to Avoid

- No default Streamlit blue anywhere
- No rainbow color schemes
- No Plotly default blue bars
- No Comic Sans, Arial, Roboto, or Inter
- No st.table() — use styled st.dataframe() or custom HTML tables
- No cluttered layouts — generous whitespace is part of the design

---

## APP STRUCTURE

### Sidebar

Contains in this order:
1. App title in DM Serif Display, teal color: "Customer Retention Analytics"
2. Subtitle in small muted text: "Churn Intelligence Dashboard"
3. Divider line
4. Theme toggle: checkbox labeled "🌙 Dark Mode"
   — Use `st.session_state['dark_mode']` to persist
5. Divider line
6. Page navigation: `st.radio` with no label, options:
   - "📊 The Business Problem"
   - "🔍 Why Customers Leave"
   - "🎯 Who to Act On First"
7. Bottom of sidebar — small muted italic footer:
   "Built on telecom data · Framework applies to any subscription business"

---

### PAGE 1 — "The Business Problem"

**Purpose:** Executive summary. A client should understand the scale of the problem
in under 10 seconds. Lead with money, not percentages.

**Layout:**

Row 1 — Four metric cards side by side (st.columns(4)):
- **Total Customers** — 7,032 — icon: 👥
- **Customers Lost** — 1,869 — icon: 📉 — value in red (#EF4444)
- **Churn Rate** — 26.6% — icon: ⚠️ — value in amber (#F59E0B)
- **Monthly Revenue at Risk** — $139,131 — icon: 💸 — value in red (#EF4444)

Row 2 — Two columns (60/40 split):
- Left (60%): Donut chart — Churned vs Retained
  - Retained slice: teal (#0D9488)
  - Churned slice: red (#EF4444)
  - Center annotation: "26.6% Churn Rate"
  - Title: "Customer Retention Overview"

- Right (40%): A styled insight card (not a chart) containing:
  - Heading: "What This Means"
  - Three bullet points in clean styled list:
    - "1 in 4 customers is leaving every cycle"
    - "Month-to-month customers churn at 15× the rate of two-year contracts"
    - "The top 20% highest-risk customers account for a 2.55× concentration of churn"
  - Bottom note in muted small text:
    "Findings based on 7,032 customers · Logistic Regression · AUC 0.835"

---

### PAGE 2 — "Why Customers Leave"

**Purpose:** Show the two key drivers with statistical backing. Frame findings
as business insight, not model output.

**Layout:**

Row 1 — Section header with subtitle:
- Title: "Churn Drivers"
- Subtitle in muted text: "Two factors explain the majority of churn risk"

Row 2 — Two columns (50/50):

Left — Bar chart: Churn Rate by Contract Type
- Horizontal bars (easier to read labels)
- Bars colored by churn rate intensity (use teal-to-red gradient logic:
  low churn = teal, high churn = red — or simply use teal with opacity scaling)
- Add a vertical dashed reference line at the overall 26.6% average
- Label each bar with the churn rate percentage
- Title: "Contract Type vs Churn Rate"
- Small caption below: "Chi-square: 1179.5 · p < 0.001"

Right — Bar chart: Churn Rate by Payment Method
- Same style as contract chart
- Same reference line at 26.6%
- Title: "Payment Method vs Churn Rate"
- Small caption below: "Chi-square: 645.4 · p < 0.001"

Row 3 — Full width:
Tenure analysis — Line or area chart showing churn rate across tenure buckets
- Bin tenure into groups: 0-12 months, 13-24, 25-36, 37-48, 49-60, 61-72
- Compute churn rate per bin
- Area chart in teal with low opacity fill
- Title: "Churn Rate by Customer Tenure"
- Insight callout below chart (styled box, teal left border):
  "Newer customers churn at significantly higher rates. Customers in their
   first year are the highest-risk cohort — early engagement is critical."

---

### PAGE 3 — "Who to Act On First"

**Purpose:** This is the action page. Show the client exactly who to target and
why focusing on the top risk tier is more efficient than random outreach.

**Layout:**

Row 1 — Section header:
- Title: "Retention Targeting"
- Subtitle: "Prioritise your retention budget where it has the most impact"

Row 2 — Three columns (risk tier summary cards):
- **High Risk** card: 1,800 customers · probability > 70% · red accent
  — "Immediate personal outreach + retention offer"
- **Medium Risk** card: 1,736 customers · 40–70% · amber accent
  — "Soft engagement — loyalty rewards or check-in"
- **Low Risk** card: 3,496 customers · below 40% · green accent
  — "Routine monitoring — no immediate action needed"

Row 3 — Two columns (55/45):

Left (55%) — Bar or grouped bar chart: The 2.55× efficiency visual
- Two bars only:
  - "Random Outreach" — 26.6% churn rate — teal
  - "Top 20% Targeted" — 67.9% churn rate — deep teal or accent
- Add a annotation arrow or callout: "2.55× more churners reached"
- Title: "Targeting Efficiency: Top 20% vs Random"
- Caption: "Same budget. 2.55× the impact."

Right (45%) — Donut or stacked bar: Risk tier distribution
- High / Medium / Low using exact risk colors
- Show both count and percentage in legend
- Title: "Customer Risk Distribution"

Row 4 — Full width:
Filterable customer table
- Columns to show: Contract, PaymentMethod, tenure (display as integer),
  MonthlyCharges (format as $XX.XX), ChurnProbability (format as XX.X%),
  RiskTier (styled badge — colored text matching risk colors)
- Default filter: show only High risk customers
- Add st.selectbox filter: "Filter by Risk Tier" — All / High / Medium / Low
- Add st.selectbox filter: "Filter by Contract Type" — All / Month-to-month /
  One year / Two year
- Sort by ChurnProbability descending by default
- Show row count above table: "Showing X customers"
- Style RiskTier column as colored badges using st.dataframe with column config
  or custom HTML — do not show plain text

---

## IMPLEMENTATION NOTES FOR THE AGENT

1. **Session state initialisation** — add this at the top of app.py before any page logic:
   ```python
   if 'dark_mode' not in st.session_state:
       st.session_state['dark_mode'] = False
   ```

2. **Theme injection** — build a function `inject_theme(dark: bool)` that writes
   the full CSS block via `st.markdown(..., unsafe_allow_html=True)`. Call it once
   at the top of every page render, passing the current theme state.

3. **Caching** — wrap data loading and model inference in `@st.cache_data` to prevent
   recomputation on every interaction.

4. **Error handling** — wrap the joblib.load and CSV reads in try/except. If files
   are missing, show a clear st.error message explaining which file is missing.

5. **No hardcoded colors outside the color dictionaries** — every color reference
   in the app must come from the LIGHT, DARK, or ACCENT dictionaries. This ensures
   the theme toggle works cleanly everywhere.

6. **Chart font color** — pass the correct text color into Plotly layout:
   ```python
   theme = DARK if st.session_state['dark_mode'] else LIGHT
   fig.update_layout(font=dict(color=theme['text'], family='DM Sans'))
   ```

7. **Streamlit config** — create a `.streamlit/config.toml` file in the project
   folder with:
   ```toml
   [theme]
   base = "light"
   
   [server]
   headless = true
   ```
   This prevents Streamlit's own theme from fighting your custom CSS.

8. **requirements.txt** — create this file alongside app.py:
   ```
   streamlit>=1.32.0
   pandas>=2.0.0
   numpy>=1.24.0
   plotly>=5.18.0
   scikit-learn>=1.3.0
   joblib>=1.3.0
   ```

---

## COPY AND TONE GUIDE

All visible text in the dashboard must follow this tone:
- Business language, not data science language
- "Customers lost" not "positive churn instances"
- "Monthly revenue at risk" not "sum of MonthlyCharges where Churn=1"
- "Who to act on first" not "high probability prediction output"
- Statistical evidence (chi-square, AUC) appears only as small captions — never
  as headline claims
- The word "model" should appear at most once in the entire dashboard

---

## WHAT GOOD LOOKS LIKE

When complete, the dashboard should:
- Load in under 3 seconds on a standard laptop
- Switch between light and dark mode instantly with no layout shift
- Show no Streamlit default blue anywhere
- Have no Plotly toolbar visible on any chart
- Display all dollar amounts formatted with $ prefix and 2 decimal places
- Display all percentages formatted with 1 decimal place and % suffix
- Be navigable entirely from the sidebar
- Look professional enough that a non-technical client trusts the analyst
  who built it within 30 seconds of seeing it

---

*Spec version 1.0 — Customer Retention Analytics — Churn Intelligence Dashboard*
*Analyst: Susil Bhattarai*
