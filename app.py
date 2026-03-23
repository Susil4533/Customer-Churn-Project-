import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# Set page config
st.set_page_config(
    page_title="Customer Retention Analytics",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 1. Session state initialisation
if 'dark_mode' not in st.session_state:
    st.session_state['dark_mode'] = False

# 2. Color System
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

# 3. Theme injection
def inject_theme(dark: bool):
    theme = DARK if dark else LIGHT
    st.markdown(f"""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600&family=DM+Serif+Display&display=swap');

    html, body, [class*="css"] {{
        font-family: 'DM Sans', sans-serif;
    }}
    h1, h2, h3 {{
        font-family: 'DM Serif Display', serif;
    }}
    .stApp, .stApp > header {{
        background-color: {theme['bg']};
    }}
    [data-testid="stSidebar"] {{
        background-color: {theme['sidebar_bg']};
        border-right: 1px solid {theme['border']};
    }}
    .stCheckbox label span, .stRadio label div, label[data-baseweb="radio"] div, .stSelectbox label div {{
        color: {theme['text']};
    }}
    </style>
    """, unsafe_allow_html=True)

# Call inject_theme at the top of page render
inject_theme(st.session_state['dark_mode'])

# 4. Data Loading Logic
@st.cache_data
def load_data():
    try:
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
    except FileNotFoundError as e:
        st.error(f"Missing file: {e.filename}. Please ensure project files are in the same folder as app.py.")
        st.stop()
    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.stop()

# Load data
df, model = load_data()

# 5. App Structure - Sidebar
with st.sidebar:
    # 1. App title
    st.markdown(
        f"<h1 style='color: {ACCENT['teal']}; margin-bottom: 0;'>Customer Retention Analytics</h1>", 
        unsafe_allow_html=True
    )
    
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT
    # 2. Subtitle
    st.markdown(
        f"<p style='color: {current_theme['text_muted']}; font-size: 0.9em; margin-top: 0;'>Churn Intelligence Dashboard</p>", 
        unsafe_allow_html=True
    )
    
    # 3. Divider line
    st.divider()
    
    # 4. Theme toggle
    # Note: Streamlit reruns the script on widget state change, so this correctly updates session state
    st.checkbox("🌙 Dark Mode", key='dark_mode')
    
    # 5. Divider line
    st.divider()
    
    # 6. Page navigation
    page = st.radio(
        label="Page navigation",
        options=[
            "📊 The Business Problem", 
            "🔍 Why Customers Leave", 
            "🎯 Who to Act On First"
        ],
        label_visibility="collapsed"
    )
    
    # Spacer
    st.markdown("<br><br><br><br>", unsafe_allow_html=True)
    
    # 7. Bottom of sidebar — small muted italic footer
    st.markdown(
        f"<p style='color: {current_theme['text_muted']}; font-size: 0.8em; font-style: italic;'>Built on telecom data · Framework applies to any subscription business</p>", 
        unsafe_allow_html=True
    )

def metric_card(title, value, icon, value_color=None):
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT
    val_color = value_color if value_color else current_theme['text']
    html = f"""
    <div style="
        background-color: {current_theme['card_bg']};
        border-radius: 12px;
        border-left: 4px solid {ACCENT['teal']};
        padding: 20px 24px;
        box-shadow: {current_theme['card_shadow']};
        margin-bottom: 20px;
    ">
        <p style="color: {current_theme['text_muted']}; font-family: 'DM Sans', sans-serif; font-size: 14px; margin-bottom: 8px;">{icon} {title}</p>
        <p style="font-family: 'DM Serif Display', serif; color: {val_color}; font-size: 32px; font-weight: bold; margin: 0;">{value}</p>
    </div>
    """
    st.markdown(html, unsafe_allow_html=True)

if page == "📊 The Business Problem":
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT
    st.markdown(f"<h2 style='color: {current_theme['text']}; margin-bottom: 0;'>The Business Problem</h2>", unsafe_allow_html=True)
    
    # Calculate metrics
    total_customers = len(df)
    churned_customers = len(df[df['Churn'] == 1])
    churn_rate = (churned_customers / total_customers) * 100
    rev_risk = df[df['Churn'] == 1]['MonthlyCharges'].sum()

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Row 1 - Cards
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        metric_card("Total Customers", f"{total_customers:,.0f}", "👥")
    with c2:
        metric_card("Customers Lost", f"{churned_customers:,.0f}", "📉", value_color=ACCENT['high'])
    with c3:
        metric_card("Churn Rate", f"{churn_rate:.1f}%", "⚠️", value_color=ACCENT['medium'])
    with c4:
        metric_card("Monthly Revenue at Risk", f"${rev_risk:,.2f}", "💸", value_color=ACCENT['high'])

    # Row 2 (60/40)
    st.markdown("<br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([6, 4])
    
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT

    with c_left:
        # Donut chart
        retained = total_customers - churned_customers
        labels = ['Retained', 'Churned']
        values = [retained, churned_customers]
        colors = [ACCENT['teal'], ACCENT['high']]

        fig = go.Figure(data=[go.Pie(
            labels=labels, 
            values=values, 
            hole=0.6,
            marker_colors=colors,
            textinfo='none',
            hoverinfo='label+percent'
        )])
        
        # Center annotation
        fig.add_annotation(
            text=f"<span style='font-family: DM Serif Display;'>{churn_rate:.1f}%<br>Churn Rate</span>",
            x=0.5, y=0.5,
            font=dict(size=20, color=current_theme['text']),
            showarrow=False
        )

        fig.update_layout(
            title=dict(
                text="Customer Retention Overview",
                font=dict(family="DM Serif Display", size=24, color=current_theme['text']),
                x=0.0
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=60, b=20, l=0, r=0),
            legend=dict(
                font=dict(family="DM Sans", color=current_theme['text']),
                orientation="h",
                yanchor="bottom",
                y=-0.1,
                xanchor="center",
                x=0.5
            )
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        
    with c_right:
        st.markdown("<br><br>", unsafe_allow_html=True) # spacing to align with chart
        insight_html = f"""
        <div style="
            background-color: {current_theme['card_bg']};
            border-radius: 12px;
            border-left: 4px solid {ACCENT['teal']};
            padding: 24px 32px;
            box-shadow: {current_theme['card_shadow']};
            height: 100%;
        ">
            <h3 style="margin-top: 0; color: {current_theme['text']}; font-family: 'DM Serif Display', serif;">What This Means</h3>
            <ul style="color: {current_theme['text']}; font-family: 'DM Sans', sans-serif; font-size: 16px; line-height: 1.6; padding-left: 20px;">
                <li style="margin-bottom: 12px;">1 in 4 customers is leaving every cycle</li>
                <li style="margin-bottom: 12px;">Month-to-month customers churn at 15× the rate of two-year contracts</li>
                <li style="margin-bottom: 12px;">The top 20% highest-risk customers account for a 2.55× concentration of churn</li>
            </ul>
            <p style="color: {current_theme['text_muted']}; font-family: 'DM Sans', sans-serif; font-size: 13px; font-style: italic; margin-top: 32px; margin-bottom: 0;">
                Findings based on {total_customers:,.0f} customers · Logistic Regression · AUC 0.835
            </p>
        </div>
        """
        st.markdown(insight_html, unsafe_allow_html=True)

elif page == "🔍 Why Customers Leave":
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT
    st.markdown(f"<h2 style='color: {current_theme['text']}; margin-bottom: 0;'>Churn Drivers</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: {current_theme['text_muted']}; font-size: 16px; margin-top: 5px;'>Two factors explain the majority of churn risk</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    overall_churn_rate = (df['Churn'].sum() / len(df)) * 100
    
    c1, c2 = st.columns(2)
    
    def plot_horizontal_bar(data, x_col, y_col, title, caption):
        data = data.sort_values(x_col, ascending=True)
        fig = px.bar(
            data, 
            y=y_col, 
            x=x_col, 
            orientation='h',
            text=data[x_col].apply(lambda x: f"{x:.1f}%"),
            color=x_col,
            color_continuous_scale=[ACCENT['teal'], ACCENT['high']],
            range_color=[0, max(data[x_col].max(), 50)]
        )
        
        fig.add_vline(x=overall_churn_rate, line_dash="dash", line_color=current_theme['text_muted'], 
                      annotation_text=f"Avg: {overall_churn_rate:.1f}%", annotation_position="top right")
        
        fig.update_traces(textposition='outside')
        fig.update_layout(
            title=dict(text=title, font=dict(family="DM Serif Display", size=20, color=current_theme['text'])),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans", color=current_theme['text']),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False, title=""),
            yaxis=dict(showgrid=False, title=""),
            coloraxis_showscale=False,
            margin=dict(l=0, r=40, t=50, b=0)
        )
        st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<p style='color: {current_theme['text_muted']}; font-size: 12px; font-style: italic; margin-top: -15px;'>{caption}</p>", unsafe_allow_html=True)

    with c1:
        contract_churn = df.groupby('Contract')['Churn'].mean().reset_index()
        contract_churn['Churn'] = contract_churn['Churn'] * 100
        plot_horizontal_bar(contract_churn, 'Churn', 'Contract', "Contract Type vs Churn Rate", "Chi-square: 1179.5 · p < 0.001")
        
    with c2:
        payment_churn = df.groupby('PaymentMethod')['Churn'].mean().reset_index()
        payment_churn['Churn'] = payment_churn['Churn'] * 100
        plot_horizontal_bar(payment_churn, 'Churn', 'PaymentMethod', "Payment Method vs Churn Rate", "Chi-square: 645.4 · p < 0.001")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Tenure Area Chart
    bins = [0, 12, 24, 36, 48, 60, 72]
    labels = ['0-12 months', '13-24', '25-36', '37-48', '49-60', '61-72']
    df['TenureBin'] = pd.cut(df['tenure'], bins=bins, labels=labels, include_lowest=True, right=True)
    
    tenure_churn = df.groupby('TenureBin', observed=True)['Churn'].mean().reset_index()
    tenure_churn['Churn'] = tenure_churn['Churn'] * 100
    
    fig2 = px.area(tenure_churn, x='TenureBin', y='Churn')
    fig2.update_traces(line_color=ACCENT['teal'], fillcolor='rgba(13, 148, 136, 0.2)')
    
    fig2.update_layout(
        title=dict(text="Churn Rate by Customer Tenure", font=dict(family="DM Serif Display", size=22, color=current_theme['text'])),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(family="DM Sans", color=current_theme['text']),
        xaxis=dict(showgrid=False, title=""),
        yaxis=dict(showgrid=False, title="", ticksuffix="%"),
        margin=dict(l=0, r=0, t=50, b=0)
    )
    st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})
    
    insight_html_2 = f"""
    <div style="
        background-color: {current_theme['card_bg']};
        border-radius: 8px;
        border-left: 4px solid {ACCENT['teal']};
        padding: 16px 24px;
        box-shadow: {current_theme['card_shadow']};
        margin-top: 20px;
    ">
        <p style="color: {current_theme['text']}; font-family: 'DM Sans', sans-serif; font-size: 16px; margin: 0; line-height: 1.5;">
            Newer customers churn at significantly higher rates. Customers in their first year are the highest-risk cohort — early engagement is critical.
        </p>
    </div>
    """
    st.markdown(insight_html_2, unsafe_allow_html=True)

elif page == "🎯 Who to Act On First":
    current_theme = DARK if st.session_state['dark_mode'] else LIGHT
    st.markdown(f"<h2 style='color: {current_theme['text']}; margin-bottom: 0;'>Retention Targeting</h2>", unsafe_allow_html=True)
    
    st.markdown(f"<p style='color: {current_theme['text_muted']}; font-size: 16px; margin-top: 5px;'>Prioritise your retention budget where it has the most impact</p>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    
    high_count = len(df[df['RiskTier'] == 'High'])
    med_count = len(df[df['RiskTier'] == 'Medium'])
    low_count = len(df[df['RiskTier'] == 'Low'])
    
    def tier_card(title, count, prob_text, accent_color, bg_tint, action_text):
        html = f"""
        <div style="
            background-color: {current_theme['card_bg']};
            border-radius: 12px;
            border-top: 4px solid {accent_color};
            padding: 20px 24px;
            box-shadow: {current_theme['card_shadow']};
            height: 100%;
        ">
            <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 12px;">
                <h3 style="margin: 0; color: {current_theme['text']}; font-family: 'DM Serif Display', serif; font-size: 20px;">{title}</h3>
                <span style="background-color: {bg_tint}; color: {accent_color}; padding: 4px 12px; border-radius: 999px; font-size: 12px; font-weight: bold;">{prob_text}</span>
            </div>
            <p style="font-family: 'DM Serif Display', serif; color: {current_theme['text']}; font-size: 32px; font-weight: bold; margin: 0 0 16px 0;">{count:,.0f} <span style="font-family: 'DM Sans', sans-serif; font-size: 14px; font-weight: normal; color: {current_theme['text_muted']};">customers</span></p>
            <p style="color: {current_theme['text_muted']}; font-size: 14px; margin: 0; line-height: 1.4;">
                <strong style="color: {current_theme['text']}">Action:</strong> {action_text}
            </p>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        tier_card("High Risk", high_count, "> 70%", ACCENT['high'], ACCENT['high_bg'], "Immediate personal outreach + retention offer")
    with c2:
        tier_card("Medium Risk", med_count, "40–70%", ACCENT['medium'], ACCENT['medium_bg'], "Soft engagement — loyalty rewards or check-in")
    with c3:
        tier_card("Low Risk", low_count, "< 40%", ACCENT['low'], ACCENT['low_bg'], "Routine monitoring — no immediate action needed")
        
    st.markdown("<br><br>", unsafe_allow_html=True)
    c_left, c_right = st.columns([55, 45])
    
    with c_left:
        baseline = (df['Churn'].sum() / len(df)) * 100
        n_top20 = int(len(df) * 0.2)
        top20_df = df.nlargest(n_top20, 'ChurnProbability')
        top20_rate = (top20_df['Churn'].sum() / len(top20_df)) * 100
        
        eff_data = pd.DataFrame({
            'Strategy': ['Random Outreach', 'Top 20% Targeted'],
            'Churn Rate': [baseline, top20_rate],
            'Color': [ACCENT['teal'], ACCENT['teal_hover']]
        })
        
        fig3 = px.bar(
            eff_data, 
            x='Strategy', 
            y='Churn Rate', 
            text=eff_data['Churn Rate'].apply(lambda x: f"{x:.1f}%")
        )
        
        fig3.update_traces(marker_color=eff_data['Color'], textposition='outside')
        
        fig3.add_annotation(
            x='Top 20% Targeted', 
            y=top20_rate,
            text=f"{(top20_rate/baseline):.2f}× more<br>churners reached",
            showarrow=True,
            arrowhead=2,
            arrowsize=1.5,
            arrowcolor=current_theme['text'],
            ax=-60,
            ay=-40,
            font=dict(size=14, color=current_theme['text'], family="DM Sans")
        )
        
        fig3.update_layout(
            title=dict(text="Targeting Efficiency: Top 20% vs Random", font=dict(family="DM Serif Display", size=20, color=current_theme['text'])),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(family="DM Sans", color=current_theme['text']),
            xaxis=dict(showgrid=False, title="", tickfont=dict(size=14)),
            yaxis=dict(showgrid=False, title="", showticklabels=False),
            margin=dict(l=0, r=0, t=50, b=0)
        )
        st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})
        st.markdown(f"<p style='color: {current_theme['text_muted']}; font-size: 13px; font-style: italic; margin-top: -15px; text-align: center;'>Same budget. {(top20_rate/baseline):.2f}× the impact.</p>", unsafe_allow_html=True)
        
    with c_right:
        tier_counts = [high_count, med_count, low_count]
        tier_labels = ['High', 'Medium', 'Low']
        tier_colors = [ACCENT['high'], ACCENT['medium'], ACCENT['low']]
        
        custom_labels = [f"{l} ({c:,.0f})" for l, c in zip(tier_labels, tier_counts)]
        
        fig4 = go.Figure(data=[go.Pie(
            labels=custom_labels, 
            values=tier_counts, 
            hole=0.6,
            marker_colors=tier_colors,
            textinfo='percent',
            hoverinfo='label+percent',
            sort=False
        )])
        
        fig4.update_layout(
            title=dict(
                text="Customer Risk Distribution",
                font=dict(family="DM Serif Display", size=20, color=current_theme['text']),
                x=0.0
            ),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            margin=dict(t=50, b=20, l=0, r=0),
            legend=dict(
                font=dict(family="DM Sans", color=current_theme['text']),
                orientation="v",
                yanchor="middle",
                y=0.5,
                xanchor="left",
                x=1.0
            )
        )
        st.plotly_chart(fig4, use_container_width=True, config={'displayModeBar': False})

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='margin-bottom: 20px; color: {current_theme['text']}; font-family: \"DM Serif Display\", serif;'>Customer Targeting List</h3>", unsafe_allow_html=True)
    
    f1, f2 = st.columns(2)
    with f1:
        risk_filter = st.selectbox("Filter by Risk Tier", ["High", "Medium", "Low", "All"])
    with f2:
        contracts = ["All"] + list(df['Contract'].unique())
        contract_filter = st.selectbox("Filter by Contract Type", contracts)
        
    filtered_df = df.copy()
    if risk_filter != "All":
        filtered_df = filtered_df[filtered_df['RiskTier'] == risk_filter]
    if contract_filter != "All":
        filtered_df = filtered_df[filtered_df['Contract'] == contract_filter]
        
    filtered_df = filtered_df.sort_values(by="ChurnProbability", ascending=False)
    
    st.markdown(f"<p style='color: {current_theme['text_muted']}; font-size: 14px;'>Showing {len(filtered_df):,.0f} customers</p>", unsafe_allow_html=True)
    
    disp_df = filtered_df[['Contract', 'PaymentMethod', 'tenure', 'MonthlyCharges', 'ChurnProbability', 'RiskTier']].copy()
    
    disp_df['ChurnProbability'] = (disp_df['ChurnProbability'] * 100).map("{:.1f}%".format)
    disp_df['MonthlyCharges'] = disp_df['MonthlyCharges'].map("${:,.2f}".format)
    disp_df['tenure'] = disp_df['tenure'].astype(int)
    
    def color_risk(val):
        if val == 'High':
            return f'color: {ACCENT["high"]}; background-color: {ACCENT["high_bg"]}; font-weight: bold; border-radius: 4px;'
        elif val == 'Medium':
            return f'color: {ACCENT["medium"]}; background-color: {ACCENT["medium_bg"]}; font-weight: bold; border-radius: 4px;'
        elif val == 'Low':
            return f'color: {ACCENT["low"]}; background-color: {ACCENT["low_bg"]}; font-weight: bold; border-radius: 4px;'
        return ''
        
    try:
        styled_df = disp_df.style.map(color_risk, subset=['RiskTier'])
    except AttributeError:
        # Fallback for pandas ver < 2.1
        styled_df = disp_df.style.applymap(color_risk, subset=['RiskTier'])
        
    st.dataframe(styled_df, use_container_width=True, hide_index=True)
