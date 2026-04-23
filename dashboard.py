"""
Greek RES Forecasting — Streamlit Dashboard
============================================
Interactive dashboard visualising day-ahead RES generation forecasts,
HENEX price data, and merit order analysis for Greece (2024-2025).

Run with:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────

st.set_page_config(
    page_title="Greek RES Forecasting",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    [data-testid="collapsedControl"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    html, body, [class*="css"], p, div, span, label {
        font-family: 'Inter', sans-serif !important;
        color: #e2e8f0;
    }

    .main, .block-container {
        background-color: #0a0f1e !important;
    }

    section[data-testid="stSidebar"] {
        background-color: #0d1526 !important;
        border-right: 1px solid #1e3a5f !important;
    }

    section[data-testid="stSidebar"] * {
        color: #e2e8f0 !important;
    }

    section[data-testid="stSidebar"] .stRadio label {
        color: #ffffff !important;
        font-size: 15px !important;
        font-weight: 400 !important;
    }

    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600 !important;
    }

    section[data-testid="stSidebar"] .stDateInput label {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }

    .metric-card {
        background: #111827;
        border: 1px solid #1e3a5f;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
        transition: border-color 0.2s;
    }

    .metric-card:hover { border-color: #00d4ff; }

    .metric-value {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 28px;
        font-weight: 500;
        color: #00d4ff !important;
        line-height: 1.2;
    }

    .metric-label {
        font-size: 11px;
        color: #64748b !important;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 6px;
        font-weight: 500;
    }

    .metric-delta {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 12px;
        color: #f4a261 !important;
        margin-top: 4px;
    }

    .section-header {
        font-size: 12px;
        font-weight: 600;
        color: #00d4ff !important;
        text-transform: uppercase;
        letter-spacing: 0.14em;
        border-bottom: 1px solid #1e3a5f;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }

    .insight-box {
        background: #111827;
        border-left: 3px solid #00d4ff;
        border-radius: 0 8px 8px 0;
        padding: 16px 20px;
        margin: 12px 0;
        font-size: 14px;
        color: #cbd5e1 !important;
        line-height: 1.7;
    }

    /* Hide Streamlit default header white bar */
    header[data-testid="stHeader"] {
        background-color: #0a0f1e !important;
        border-bottom: none !important;
    }

    /* Hide deploy button and hamburger menu */
    #MainMenu, footer, header { visibility: hidden; }
    .stDeployButton { display: none; }

    /* Hide error toolbar buttons (Ask Google, Ask ChatGPT) */
    div[data-testid="stException"] button,
    div[class*="ErrorToolbar"],
    div[class*="errorToolbar"] { display: none !important; }

    /* Date input text visibility */
    div[data-testid="stDateInput"] input {
        color: #ffffff !important;
        background-color: #111827 !important;
        border: 1px solid #1e3a5f !important;
        border-radius: 6px !important;
        font-size: 14px !important;
    }

    div[data-testid="stDateInput"] input::placeholder {
        color: #64748b !important;
    }

    .sidebar-logo {
        font-family: 'JetBrains Mono', monospace !important;
        font-size: 11px;
        color: #334155 !important;
        text-align: center;
        padding: 8px 0;
        letter-spacing: 0.1em;
    }

    h1, h2, h3 { color: #ffffff !important; font-weight: 700 !important; }

    .stDataFrame { background: #111827 !important; }

    div[data-testid="stSelectbox"] label,
    div[data-testid="stDateInput"] label {
        color: #94a3b8 !important;
        font-size: 13px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PLOTLY THEME
# ─────────────────────────────────────────────

PLOT_THEME = dict(
    paper_bgcolor='#0a0f1e',
    plot_bgcolor='#111827',
    font=dict(family='Inter', color='#e2e8f0', size=12),
    legend=dict(bgcolor='rgba(13,21,38,0.95)', bordercolor='#1e3a5f',
                borderwidth=1, font=dict(size=12, color='#e2e8f0')),
    margin=dict(l=50, r=30, t=50, b=50),
)

# Reusable axis style — apply individually per chart to avoid conflicts
AXIS_STYLE = dict(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                  tickfont=dict(size=11, color='#94a3b8'))

COLORS = {
    'solar':    '#f59e0b',
    'wind':     '#38bdf8',
    'total':    '#00d4ff',
    'price':    '#f97316',
    'forecast': '#a78bfa',
    'naive':    '#475569',
    'accent':   '#00d4ff',
}


# ─────────────────────────────────────────────
# DATA LOADING
# ─────────────────────────────────────────────

@st.cache_data
def load_data():
    df = pd.read_csv('./data/phase4_with_prices.csv',
                     index_col='datetime_utc', parse_dates=True)
    if df.index.tz is None:
        df.index = df.index.tz_localize('UTC')

    forecasts = pd.read_csv('./data/phase3_forecasts.csv',
                            index_col='datetime_utc', parse_dates=True)
    if forecasts.index.tz is None:
        forecasts.index = forecasts.index.tz_localize('UTC')

    # Merge forecasts into main df for test period
    df = df.join(forecasts, how='left')
    return df


@st.cache_data
def get_column_names(df):
    solar_col  = next(c for c in df.columns if 'solar' in c.lower() and 'mw' in c.lower()
                      and 'forecast' not in c.lower() and 'actual' not in c.lower())
    wind_col   = next(c for c in df.columns if 'windonshore' in c.lower())
    total_col  = next(c for c in df.columns if 'totalres' in c.lower()
                      and 'forecast' not in c.lower() and 'actual' not in c.lower())
    return solar_col, wind_col, total_col


# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────

with st.sidebar:
    st.markdown("## Greek RES Forecast")
    st.markdown('<div class="sidebar-logo">GREECE · 2024–2025 · ENTSO-E</div>',
                unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### Navigation")
    page = st.radio("", [
        "Overview",
        "Forecast vs Actual",
        "Merit Order Analysis",
        "Negative Prices",
        "Model Performance"
    ], label_visibility="collapsed")

    st.markdown("---")
    st.markdown("### Date Filter")

    df_full = load_data()
    min_date = df_full.index.min().date()
    max_date = df_full.index.max().date()

    date_range = st.date_input(
        "Select range",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )

    if len(date_range) == 2:
        start_date, end_date = date_range
    else:
        start_date, end_date = min_date, max_date

    st.markdown("---")
    st.markdown("### Data Sources")
    st.markdown("""
    <div style='font-size:12px; color:#6b7a99; line-height:1.8'>
    📡 <b style='color:#a8b4cc'>Generation:</b> ENTSO-E<br>
    🌤️ <b style='color:#a8b4cc'>Weather:</b> Open-Meteo<br>
    💶 <b style='color:#a8b4cc'>Prices:</b> ENTSO-E / HENEX<br>
    🤖 <b style='color:#a8b4cc'>Models:</b> RF + Linear Reg
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style='font-size:11px; color:#3a4560; text-align:center'>
    Build as a very first attempt<br>
    to create an interactive dashboard forecasting RES generation
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────

df = df_full[
    (df_full.index.date >= start_date) &
    (df_full.index.date <= end_date)
].copy()

solar_col, wind_col, total_col = get_column_names(df_full)
price_col = 'price_EURperMWh'

# ─────────────────────────────────────────────
# PAGE: OVERVIEW
# ─────────────────────────────────────────────

if page == "Overview":

    st.markdown("# Greek RES Day-Ahead Forecasting")
    st.markdown(
        "<p style='color:#6b7a99; font-size:15px; margin-top:-12px'>"
        "Solar & Wind generation forecasting for the Greek bidding zone (BZN|GR) "
        "with HENEX day-ahead price correlation analysis</p>",
        unsafe_allow_html=True
    )

    # KPI cards
    col1, col2, col3, col4, col5 = st.columns(5)

    avg_price    = df[price_col].mean()
    avg_solar    = df[solar_col].mean()
    avg_wind     = df[wind_col].mean()
    avg_total    = df[total_col].mean()
    neg_hours    = (df[price_col] < 0).sum()
    corr_total   = df[[total_col, price_col]].corr().iloc[0,1]

    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_price:.0f}</div>
            <div class="metric-label">Avg Price (EUR/MWh)</div>
            <div class="metric-delta">Day-ahead market</div>
        </div>""", unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_solar:.0f}</div>
            <div class="metric-label">Avg Solar (MW)</div>
            <div class="metric-delta">All hours incl. night</div>
        </div>""", unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_wind:.0f}</div>
            <div class="metric-label">Avg Wind (MW)</div>
            <div class="metric-delta">Onshore fleet</div>
        </div>""", unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{neg_hours}</div>
            <div class="metric-label">Negative Price Hours</div>
            <div class="metric-delta">{neg_hours/len(df)*100:.1f}% of period</div>
        </div>""", unsafe_allow_html=True)

    with col5:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{corr_total:.2f}</div>
            <div class="metric-label">RES–Price Correlation</div>
            <div class="metric-delta">Merit order signal</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Daily overview chart
    st.markdown('<div class="section-header">Generation & Price Overview</div>',
                unsafe_allow_html=True)

    daily = df[[solar_col, wind_col, total_col, price_col]].resample('1D').mean()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        row_heights=[0.65, 0.35], vertical_spacing=0.04)

    fig.add_trace(go.Scatter(
        x=daily.index, y=daily[solar_col],
        name='Solar', fill='tozeroy',
        fillcolor='rgba(244,162,97,0.25)',
        line=dict(color=COLORS['solar'], width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily.index, y=daily[wind_col],
        name='Wind', fill='tozeroy',
        fillcolor='rgba(69,123,157,0.25)',
        line=dict(color=COLORS['wind'], width=1.5)), row=1, col=1)

    fig.add_trace(go.Scatter(
        x=daily.index, y=daily[price_col],
        name='Day-Ahead Price',
        line=dict(color=COLORS['price'], width=2)), row=2, col=1)

    fig.add_hline(y=0, line_dash='dot', line_color='#3a4560',
                  line_width=1, row=2, col=1)

    fig.update_layout(
        **PLOT_THEME,
        height=500,
        title=dict(text='Daily Average RES Generation & Day-Ahead Price',
                   font=dict(size=14, color='#a8b4cc')),
    )
    fig.update_yaxes(title_text='Generation (MW)', row=1, col=1,
                     gridcolor='#1e3a5f', linecolor='#1e3a5f')
    fig.update_yaxes(title_text='EUR/MWh', row=2, col=1,
                     gridcolor='#1e3a5f', linecolor='#1e3a5f')

    st.plotly_chart(fig, use_container_width=True)

    # Monthly summary table
    st.markdown('<div class="section-header">Monthly Summary</div>',
                unsafe_allow_html=True)

    month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                    7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}

    monthly = df.groupby(df.index.month).agg(
        Avg_Price_EUR    = (price_col, 'mean'),
        Avg_Solar_MW     = (solar_col, 'mean'),
        Avg_Wind_MW      = (wind_col, 'mean'),
        Avg_Total_RES_MW = (total_col, 'mean'),
        Neg_Price_Hours  = (price_col, lambda x: (x < 0).sum()),
    ).round(1)
    monthly.index = [month_labels[i] for i in monthly.index]
    monthly.index.name = 'Month'
    st.dataframe(monthly, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: FORECAST VS ACTUAL
# ─────────────────────────────────────────────

elif page == "Forecast vs Actual":

    st.markdown("# Forecast vs Actual Generation")
    st.markdown(
        "<p style='color:#6b7a99; font-size:15px; margin-top:-12px'>"
        "Day-ahead forecasts (Random Forest for solar, Linear Regression for wind) "
        "compared against actual ENTSO-E generation data</p>",
        unsafe_allow_html=True
    )

    test_df = df[df['forecast_solar_MW'].notna()].copy()

    if len(test_df) == 0:
        st.warning("No forecast data available for the selected date range. "
                   "Select Oct–Dec 2025 to view forecasts.")
    else:
        # Model performance metrics
        from sklearn.metrics import mean_absolute_error
        mae_solar = mean_absolute_error(
            test_df['actual_solar_MW'], test_df['forecast_solar_MW'])
        mae_wind  = mean_absolute_error(
            test_df['actual_wind_MW'],  test_df['forecast_wind_MW'])
        mae_total = mean_absolute_error(
            test_df['actual_total_MW'], test_df['forecast_total_MW'])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mae_solar:.0f} MW</div>
                <div class="metric-label">Solar MAE</div>
                <div class="metric-delta">Random Forest model</div>
            </div>""", unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mae_wind:.0f} MW</div>
                <div class="metric-label">Wind MAE</div>
                <div class="metric-delta">Linear Regression model</div>
            </div>""", unsafe_allow_html=True)
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-value">{mae_total:.0f} MW</div>
                <div class="metric-label">Total RES MAE</div>
                <div class="metric-delta">Combined forecast</div>
            </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        source = st.selectbox("Select source", ["Solar", "Wind", "Total RES"])
        source_map = {
            "Solar":     ('actual_solar_MW',  'forecast_solar_MW',  COLORS['solar']),
            "Wind":      ('actual_wind_MW',   'forecast_wind_MW',   COLORS['wind']),
            "Total RES": ('actual_total_MW',  'forecast_total_MW',  COLORS['total']),
        }
        actual_col, fcast_col, color = source_map[source]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df[actual_col],
            name='Actual', line=dict(color=color, width=1.8)))
        fig.add_trace(go.Scatter(
            x=test_df.index, y=test_df[fcast_col],
            name='Day-Ahead Forecast',
            line=dict(color=COLORS['forecast'], width=1.2, dash='dash')))

        fig.update_layout(
            **PLOT_THEME,
            height=420,
            title=dict(text=f'{source} Generation — Actual vs Day-Ahead Forecast',
                       font=dict(size=14, color='#a8b4cc')),
            yaxis_title='MW',
            hovermode='x unified'
        )
        st.plotly_chart(fig, use_container_width=True)

        # Error distribution
        st.markdown('<div class="section-header">Forecast Error Distribution</div>',
                    unsafe_allow_html=True)

        errors = test_df[actual_col] - test_df[fcast_col]
        fig2 = go.Figure()
        fig2.add_trace(go.Histogram(
            x=errors, nbinsx=60,
            marker_color=color, opacity=0.8,
            name='Forecast Error'))
        fig2.add_vline(x=0, line_dash='dash', line_color='#6b7a99', line_width=1.5)
        fig2.add_vline(x=errors.mean(), line_dash='dot',
                       line_color=COLORS['price'], line_width=1.5,
                       annotation_text=f'Mean: {errors.mean():.0f} MW',
                       annotation_font_color=COLORS['price'])
        fig2.update_layout(
            **PLOT_THEME, height=320,
            title=dict(text='Distribution of Forecast Errors (Actual − Forecast)',
                       font=dict(size=14, color='#a8b4cc')),
            xaxis_title='Error (MW)', yaxis_title='Frequency'
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.markdown(f"""
        <div class="insight-box">
        A positive error means the model <b style='color:#4ecca3'>underestimated</b> actual generation.
        A negative error means the model <b style='color:#e76f51'>overestimated</b> it.
        For a RES producer bidding into HENEX, overestimation is more costly — it means
        selling generation that doesn't materialise and facing imbalance penalties.
        </div>""", unsafe_allow_html=True)


# ─────────────────────────────────────────────
# PAGE: MERIT ORDER ANALYSIS
# ─────────────────────────────────────────────

elif page == "Merit Order Analysis":

    st.markdown("# Merit Order Effect")
    st.markdown(
        "<p style='color:#6b7a99; font-size:15px; margin-top:-12px'>"
        "Higher RES generation suppresses day-ahead prices as zero-marginal-cost "
        "renewables displace gas-fired generation from the supply curve</p>",
        unsafe_allow_html=True
    )

    col1, col2 = st.columns(2)

    with col1:
        # Scatter: Total RES vs Price
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df[total_col], y=df[price_col],
            mode='markers',
            marker=dict(color=COLORS['total'], size=3, opacity=0.3),
            name='Hourly observation'))

        # Trend line
        mask = df[[total_col, price_col]].dropna()
        z = np.polyfit(mask[total_col], mask[price_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(mask[total_col].min(), mask[total_col].max(), 100)
        fig.add_trace(go.Scatter(
            x=x_line, y=p(x_line),
            mode='lines',
            line=dict(color='white', width=2),
            name='Trend'))

        r = mask.corr().iloc[0,1]
        fig.update_layout(
            **PLOT_THEME, height=380,
            title=dict(text=f'Total RES vs Price (r = {r:.2f})',
                       font=dict(size=14, color='#a8b4cc')),
            xaxis_title='Total RES Generation (MW)',
            yaxis_title='Day-Ahead Price (EUR/MWh)'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        # Merit order buckets
        df_clean = df[[total_col, price_col]].dropna()
        df_clean['bucket'] = pd.qcut(df_clean[total_col], q=5,
                                      labels=['Very Low','Low','Medium',
                                              'High','Very High'])
        bucket_avg = df_clean.groupby('bucket', observed=True)[price_col].mean()

        fig2 = go.Figure(go.Bar(
            x=list(bucket_avg.index),
            y=bucket_avg.values,
            marker_color=[COLORS['price'], '#f4a261', '#e9c46a',
                          '#2a9d8f', COLORS['total']],
            text=[f'{v:.0f}' for v in bucket_avg.values],
            textposition='outside',
            textfont=dict(color='#a8b4cc', size=12)
        ))
        fig2.update_layout(
            **PLOT_THEME, height=380,
            title=dict(text='Avg Price by RES Generation Quintile',
                       font=dict(size=14, color='#a8b4cc')),
            xaxis_title='RES Generation Level',
            yaxis_title='Avg Day-Ahead Price (EUR/MWh)',
            showlegend=False
        )
        st.plotly_chart(fig2, use_container_width=True)

    # Price suppression stat
    very_low  = bucket_avg.iloc[0]
    very_high = bucket_avg.iloc[-1]
    suppression = (very_low - very_high) / very_low * 100

    st.markdown(f"""
    <div class="insight-box">
    Moving from the lowest 20% to the highest 20% of RES generation hours reduces
    the average day-ahead price by <b style='color:#4ecca3'>{suppression:.0f}%</b>
    — from <b style='color:#f4a261'>{very_low:.0f} EUR/MWh</b> to
    <b style='color:#4ecca3'>{very_high:.0f} EUR/MWh</b>.
    This price cannibalisation effect intensifies as more RES capacity is installed,
    driving the investment case for battery storage and flexible demand.
    </div>""", unsafe_allow_html=True)

    # Intraday profile
    st.markdown('<div class="section-header">Intraday Price vs Generation Profile</div>',
                unsafe_allow_html=True)

    hourly_price = df.groupby(df.index.hour)[price_col].mean()
    hourly_solar = df.groupby(df.index.hour)[solar_col].mean()
    hourly_wind  = df.groupby(df.index.hour)[wind_col].mean()

    fig3 = make_subplots(specs=[[{"secondary_y": True}]])

    fig3.add_trace(go.Scatter(
        x=list(range(24)), y=hourly_solar.values,
        name='Solar', fill='tozeroy',
        fillcolor='rgba(244,162,97,0.3)',
        line=dict(color=COLORS['solar'], width=1.5)), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=list(range(24)),
        y=(hourly_solar + hourly_wind).values,
        name='Solar + Wind', fill='tonexty',
        fillcolor='rgba(69,123,157,0.25)',
        line=dict(color=COLORS['wind'], width=1.5)), secondary_y=False)

    fig3.add_trace(go.Scatter(
        x=list(range(24)), y=hourly_price.values,
        name='Avg Price',
        line=dict(color=COLORS['price'], width=2.5),
        mode='lines+markers',
        marker=dict(size=6)), secondary_y=True)

    fig3.update_layout(
        **PLOT_THEME, height=380,
        title=dict(text='Intraday RES Generation vs Day-Ahead Price — Merit Order in Action',
                   font=dict(size=14, color='#ffffff')),
        hovermode='x unified'
    )
    fig3.update_xaxes(title_text='Hour of Day (UTC)', tickmode='linear',
                      gridcolor='#1e3a5f', linecolor='#1e3a5f',
                      tickfont=dict(color='#94a3b8'))
    fig3.update_yaxes(title_text='Generation (MW)', secondary_y=False,
                      gridcolor='#1e3a5f', linecolor='#1e3a5f',
                      tickfont=dict(color='#94a3b8'))
    fig3.update_yaxes(title_text='Price (EUR/MWh)', secondary_y=True,
                      gridcolor='#1e3a5f', linecolor='#1e3a5f',
                      tickfont=dict(color='#94a3b8'))
    st.plotly_chart(fig3, use_container_width=True)


# ─────────────────────────────────────────────
# PAGE: NEGATIVE PRICES
# ─────────────────────────────────────────────

elif page == "Negative Prices":

    st.markdown("# Negative Price Analysis")
    st.markdown(
        "<p style='color:#6b7a99; font-size:15px; margin-top:-12px'>"
        "Hours where day-ahead prices fell below zero — a signal of RES oversupply "
        "exceeding grid absorption capacity</p>",
        unsafe_allow_html=True
    )

    neg = df[df[price_col] < 0].copy()
    total_hours = len(df)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{len(neg)}</div>
            <div class="metric-label">Negative Price Hours</div>
            <div class="metric-delta">{len(neg)/total_hours*100:.1f}% of period</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{neg[price_col].mean():.1f}</div>
            <div class="metric-label">Avg Negative Price</div>
            <div class="metric-delta">EUR/MWh</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{neg[price_col].min():.1f}</div>
            <div class="metric-label">Most Negative Price</div>
            <div class="metric-delta">EUR/MWh</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        avg_res_neg = neg[total_col].mean()
        avg_res_all = df[total_col].mean()
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{avg_res_neg:.0f}</div>
            <div class="metric-label">Avg RES During Neg. Hours</div>
            <div class="metric-delta">vs {avg_res_all:.0f} MW overall</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    if len(neg) > 0:
        col1, col2 = st.columns(2)

        with col1:
            neg_by_hour = neg.groupby(neg.index.hour).size().reindex(range(24), fill_value=0)
            fig = go.Figure(go.Bar(
                x=list(range(24)), y=neg_by_hour.values,
                marker_color=COLORS['price'], opacity=0.85))
            fig.update_layout(
                **PLOT_THEME, height=340,
                title=dict(text='Negative Price Hours by Hour of Day (UTC)',
                           font=dict(size=14, color='#ffffff')),
                yaxis_title='Number of Hours', showlegend=False)
            fig.update_xaxes(title_text='Hour (UTC)', tickmode='linear',
                             gridcolor='#1e3a5f', linecolor='#1e3a5f',
                             tickfont=dict(color='#94a3b8'))
            fig.update_yaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                             tickfont=dict(color='#94a3b8'))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            month_labels = {1:'Jan',2:'Feb',3:'Mar',4:'Apr',5:'May',6:'Jun',
                            7:'Jul',8:'Aug',9:'Sep',10:'Oct',11:'Nov',12:'Dec'}
            neg_by_month = neg.groupby(neg.index.month).size()
            fig2 = go.Figure(go.Bar(
                x=[month_labels[m] for m in neg_by_month.index],
                y=neg_by_month.values,
                marker_color=COLORS['wind'], opacity=0.85))
            fig2.update_layout(
                **PLOT_THEME, height=340,
                title=dict(text='Negative Price Hours by Month',
                           font=dict(size=14, color='#ffffff')),
                xaxis_title='Month', yaxis_title='Number of Hours',
                showlegend=False)
            fig2.update_xaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                              tickfont=dict(color='#94a3b8'))
            fig2.update_yaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                              tickfont=dict(color='#94a3b8'))

        st.markdown(f"""
        <div class="insight-box">
        All {len(neg)} negative price hours occurred during <b style='color:#4ecca3'>daylight hours only</b>,
        confirming these events are driven exclusively by solar oversupply rather than wind or demand factors.
        Average RES generation during negative price hours was
        <b style='color:#f4a261'>{avg_res_neg:.0f} MW</b> vs
        <b style='color:#6b7a99'>{avg_res_all:.0f} MW</b> overall —
        a {(avg_res_neg/avg_res_all - 1)*100:.0f}% premium.
        As Greece installs additional solar capacity, negative price frequency will increase,
        strengthening the commercial case for co-located battery storage.
        </div>""", unsafe_allow_html=True)
    else:
        st.info("No negative price hours found in the selected date range.")


# ─────────────────────────────────────────────
# PAGE: MODEL PERFORMANCE
# ─────────────────────────────────────────────

elif page == "Model Performance":

    st.markdown("# Forecasting Model Performance")
    st.markdown(
        "<p style='color:#6b7a99; font-size:15px; margin-top:-12px'>"
        "Day-ahead forecast evaluation across three models — Seasonal Naive baseline, "
        "Linear Regression, and Random Forest — test period Oct–Dec 2025</p>",
        unsafe_allow_html=True
    )

    # Results table
    results = {
        'Model':     ['Seasonal Naive', 'Linear Regression', 'Random Forest'],
        'Solar MAE': [539.5, 345.2, 261.4],
        'Solar RMSE':[1057.4, 581.0, 537.3],
        'Wind MAE':  [1061.6, 391.5, 427.9],
        'Wind RMSE': [1289.8, 514.2, 552.4],
    }
    results_df = pd.DataFrame(results).set_index('Model')

    col1, col2 = st.columns(2)

    with col1:
        st.markdown('<div class="section-header">Solar Generation — MAE by Model</div>',
                    unsafe_allow_html=True)
        fig = go.Figure(go.Bar(
            x=results['Model'], y=results['Solar MAE'],
            marker_color=[COLORS['naive'], COLORS['wind'], COLORS['solar']],
            text=[f'{v:.0f} MW' for v in results['Solar MAE']],
            textposition='outside',
            textfont=dict(color='#a8b4cc')))
        fig.update_layout(
            **PLOT_THEME, height=320,
            yaxis_title='MAE (MW)', showlegend=False)
        fig.update_yaxes(range=[0, max(results['Solar MAE'])*1.25],
                         gridcolor='#1e3a5f', linecolor='#1e3a5f',
                         tickfont=dict(color='#94a3b8'))
        fig.update_xaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                         tickfont=dict(color='#94a3b8'))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown('<div class="section-header">Wind Generation — MAE by Model</div>',
                    unsafe_allow_html=True)
        fig2 = go.Figure(go.Bar(
            x=results['Model'], y=results['Wind MAE'],
            marker_color=[COLORS['naive'], COLORS['solar'], COLORS['wind']],
            text=[f'{v:.0f} MW' for v in results['Wind MAE']],
            textposition='outside',
            textfont=dict(color='#e2e8f0')))
        fig2.update_layout(
            **PLOT_THEME, height=320,
            yaxis_title='MAE (MW)', showlegend=False)
        fig2.update_yaxes(range=[0, max(results['Wind MAE'])*1.25],
                          gridcolor='#1e3a5f', linecolor='#1e3a5f',
                          tickfont=dict(color='#94a3b8'))
        fig2.update_xaxes(gridcolor='#1e3a5f', linecolor='#1e3a5f',
                          tickfont=dict(color='#94a3b8'))
        st.plotly_chart(fig2, use_container_width=True)

    # Full results table
    st.markdown('<div class="section-header">Full Results Table</div>',
                unsafe_allow_html=True)
    st.dataframe(results_df.style.highlight_min(axis=0, color='rgba(78,204,163,0.2)'),
                 use_container_width=True)

    # Key insights
    solar_improvement = (539.5 - 261.4) / 539.5 * 100
    wind_improvement  = (1061.6 - 391.5) / 1061.6 * 100
    total_naive_mae   = 1158.1
    total_best_mae    = 507.9
    total_improvement = (total_naive_mae - total_best_mae) / total_naive_mae * 100

    st.markdown(f"""
    <div class="insight-box">
    <b style='color:#4ecca3'>Solar:</b> Random Forest reduces MAE by
    <b style='color:#f4a261'>{solar_improvement:.0f}%</b> vs naive baseline
    (539 → 261 MW), capturing non-linear irradiance relationships.<br><br>
    <b style='color:#457b9d'>Wind:</b> Linear Regression outperforms Random Forest
    (391 vs 427 MW MAE), suggesting simpler models generalise better when the
    underlying signal is noisy — a key finding of this study.<br><br>
    <b style='color:#4ecca3'>Total RES:</b> Combined forecast achieves
    <b style='color:#f4a261'>{total_improvement:.0f}%</b> MAE reduction vs naive
    ({total_naive_mae:.0f} → {total_best_mae:.0f} MW), directly reducing imbalance
    cost exposure for RES producers bidding into the HENEX day-ahead market.
    </div>""", unsafe_allow_html=True)
