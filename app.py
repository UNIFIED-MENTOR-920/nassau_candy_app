import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import warnings
warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Nassau Candy – Factory Reallocation & Shipping Optimization",
    page_icon="🍬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
/* ---- global ---- */
[data-testid="stAppViewContainer"] { background: #f7f8fc; }
[data-testid="stSidebar"] { background: #1a1f36; }
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stMultiSelect label,
[data-testid="stSidebar"] .stSlider label { color: #a0aec0 !important; font-size:13px; }
h1,h2,h3 { color: #1a1f36; font-family: 'Segoe UI', sans-serif; }

/* ---- KPI cards ---- */
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 20px 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08);
    border-left: 4px solid #4f46e5;
    margin-bottom: 6px;
}
.kpi-label { font-size:13px; color:#64748b; margin:0; font-weight:500; text-transform:uppercase; letter-spacing:.5px; }
.kpi-value { font-size:28px; font-weight:700; color:#1a1f36; margin:4px 0 0; }
.kpi-delta { font-size:12px; margin:2px 0 0; }
.kpi-delta.good { color:#10b981; }
.kpi-delta.bad  { color:#ef4444; }

/* ---- section cards ---- */
.section-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}

/* ---- recommendation table ---- */
.rec-table { width:100%; border-collapse:collapse; font-size:14px; }
.rec-table th { background:#f1f5f9; color:#475569; font-weight:600;
                padding:10px 14px; text-align:left; border-bottom:2px solid #e2e8f0; }
.rec-table td { padding:10px 14px; border-bottom:1px solid #f1f5f9; color:#1e293b; vertical-align:middle; }
.rec-table tr:hover td { background:#f8fafc; }

/* ---- badges ---- */
.badge { display:inline-block; padding:3px 10px; border-radius:20px; font-size:12px; font-weight:600; }
.badge-low    { background:#d1fae5; color:#065f46; }
.badge-medium { background:#fef3c7; color:#92400e; }
.badge-high   { background:#fee2e2; color:#991b1b; }
.badge-rec    { background:#ede9fe; color:#5b21b6; }
.badge-cur    { background:#e0f2fe; color:#0369a1; }

/* ---- model metric box ---- */
.model-metric { background:#f8fafc; border-radius:8px; padding:12px 16px;
                border:1px solid #e2e8f0; margin-bottom:8px; }
.model-metric .name  { font-weight:700; font-size:15px; color:#1a1f36; }
.model-metric .stats { font-size:13px; color:#64748b; margin-top:4px; }
.best-model { border:2px solid #4f46e5 !important; background:#eef2ff !important; }

/* ---- sidebar header ---- */
.sidebar-title { font-size:18px; font-weight:700; color:#fff !important;
                 margin-bottom:4px; padding:0; }
.sidebar-sub   { font-size:12px; color:#94a3b8 !important; margin-bottom:20px; }

/* ---- tab override ---- */
.stTabs [data-baseweb="tab-list"] { background:#fff; border-radius:10px;
    padding:4px; gap:4px; box-shadow:0 1px 4px rgba(0,0,0,0.06); }
.stTabs [data-baseweb="tab"] { border-radius:8px; font-weight:500; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# CONSTANTS
# ──────────────────────────────────────────────
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":   "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":           "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":      "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":          "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":   "Wicked Choccy's",
    "Laffy Taffy":                         "Sugar Shack",
    "SweeTARTS":                           "Sugar Shack",
    "Nerds":                               "Sugar Shack",
    "Fun Dip":                             "Sugar Shack",
    "Fizzy Lifting Drinks":                "Sugar Shack",
    "Everlasting Gobstopper":              "Secret Factory",
    "Hair Toffee":                         "The Other Factory",
    "Lickable Wallpaper":                  "Secret Factory",
    "Wonka Gum":                           "Secret Factory",
    "Kazookles":                           "The Other Factory",
}

FACTORY_COORDS = {
    "Lot's O' Nuts":     {"lat": 32.881893, "lon": -111.768036, "location": "Arizona"},
    "Wicked Choccy's":   {"lat": 32.076176, "lon": -81.088371,  "location": "Georgia"},
    "Sugar Shack":       {"lat": 48.11914,  "lon": -96.18115,   "location": "Minnesota"},
    "Secret Factory":    {"lat": 41.446333, "lon": -90.565487,  "location": "Illinois"},
    "The Other Factory": {"lat": 35.1175,   "lon": -89.971107,  "location": "Tennessee"},
}

# Factory-region distance penalty (days)
FACTORY_REGION_DIST = {
    "Lot's O' Nuts":     {"Interior": 1, "Atlantic": 2, "Gulf": 1, "Pacific": 0},
    "Wicked Choccy's":   {"Interior": 1, "Atlantic": 0, "Gulf": 1, "Pacific": 2},
    "Sugar Shack":       {"Interior": 0, "Atlantic": 1, "Gulf": 1, "Pacific": 2},
    "Secret Factory":    {"Interior": 0, "Atlantic": 1, "Gulf": 1, "Pacific": 2},
    "The Other Factory": {"Interior": 0, "Atlantic": 1, "Gulf": 0, "Pacific": 1},
}

SHIP_BASE_LEAD = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 1}
FACTORIES = list(FACTORY_COORDS.keys())
REGIONS   = ["Interior", "Atlantic", "Gulf", "Pacific"]

FACTORY_COLORS = {
    "Lot's O' Nuts":     "#4f46e5",
    "Wicked Choccy's":   "#e11d48",
    "Sugar Shack":       "#059669",
    "Secret Factory":    "#d97706",
    "The Other Factory": "#7c3aed",
}
REGION_COLORS = {"Interior": "#4f46e5", "Atlantic": "#0ea5e9", "Gulf": "#10b981", "Pacific": "#f59e0b"}
DIV_COLORS    = {"Chocolate": "#92400e", "Sugar": "#ec4899", "Other": "#6366f1"}


# ──────────────────────────────────────────────
# DATA LOADING & PREPROCESSING
# ──────────────────────────────────────────────
@st.cache_data
def load_data():
    import os, io
    # Try uploaded file first
    for fname in ["Nassau_Candy_Distributor.csv"]:
        if os.path.exists(fname):
            df = pd.read_csv(fname)
            break
    else:
        st.error("❌ CSV not found. Place `Nassau_Candy_Distributor.csv` in the same folder as `app.py`.")
        st.stop()

    # Types
    for col in ["Sales", "Cost", "Gross Profit", "Units"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True, errors="coerce")

    # Factory from product
    df["Factory"] = df["Product Name"].map(PRODUCT_FACTORY)

    # Lead time = ship mode base + region distance penalty
    df["Lead Time"] = df["Ship Mode"].map(SHIP_BASE_LEAD) + df["Region"].map(
        lambda r: 0
    )
    df["Lead Time"] = df.apply(
        lambda row: SHIP_BASE_LEAD.get(row["Ship Mode"], 5)
                    + FACTORY_REGION_DIST.get(row["Factory"], {}).get(row["Region"], 1),
        axis=1,
    )

    df["Profit Margin (%)"] = (df["Gross Profit"] / df["Sales"] * 100).round(2)
    df["Year"]  = df["Order Date"].dt.year
    df["Month"] = df["Order Date"].dt.month
    df["Month Name"] = df["Order Date"].dt.strftime("%b")

    df.dropna(subset=["Factory", "Sales", "Lead Time"], inplace=True)
    return df


@st.cache_resource
def train_models(df: pd.DataFrame):
    """Train Linear Regression, Random Forest, Gradient Boosting on lead time."""
    features = ["Ship Mode", "Region", "Factory", "Division"]
    target   = "Lead Time"

    subset = df[features + [target]].dropna()
    le = {}
    X = pd.DataFrame()
    for col in features:
        enc = LabelEncoder()
        X[col] = enc.fit_transform(subset[col].astype(str))
        le[col] = enc
    y = subset[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    models = {
        "Linear Regression":       LinearRegression(),
        "Random Forest":           RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":       GradientBoostingRegressor(n_estimators=150, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        results[name] = {
            "model":  model,
            "rmse":   round(np.sqrt(mean_squared_error(y_test, preds)), 4),
            "mae":    round(mean_absolute_error(y_test, preds), 4),
            "r2":     round(r2_score(y_test, preds), 4),
            "preds":  preds,
            "y_test": y_test.values,
        }

    best = min(results, key=lambda k: results[k]["rmse"])
    return results, le, features, best


def predict_lead(model, le, features, ship_mode, region, factory, division):
    row = {"Ship Mode": ship_mode, "Region": region, "Factory": factory, "Division": division}
    X = np.array([[le[f].transform([row[f]])[0] for f in features]])
    return round(model.predict(X)[0], 2)


def badge(level):
    cls = {"low": "badge-low", "medium": "badge-medium", "high": "badge-high"}.get(level, "badge-low")
    return f'<span class="badge {cls}">{level.upper()}</span>'


# ──────────────────────────────────────────────
# SIDEBAR
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown('<p class="sidebar-title">🍬 Nassau Candy</p>', unsafe_allow_html=True)
    st.markdown('<p class="sidebar-sub">Factory Reallocation & Shipping Optimization</p>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("**📌 Global Filters**")
    sel_regions   = st.multiselect("Region",    REGIONS,   default=REGIONS)
    sel_factories = st.multiselect("Factory",   FACTORIES, default=FACTORIES)
    sel_divisions = st.multiselect("Division",  ["Chocolate", "Sugar", "Other"], default=["Chocolate", "Sugar", "Other"])
    sel_modes     = st.multiselect("Ship Mode", list(SHIP_BASE_LEAD.keys()), default=list(SHIP_BASE_LEAD.keys()))

    st.markdown("---")
    st.markdown("**⚙️ Optimization Priority**")
    priority = st.slider("Speed ◀──────▶ Profit", 0, 100, 50,
                         help="0 = maximize profit, 100 = minimize lead time")
    top_n = st.slider("Top-N Recommendations", 3, 15, 7)
    st.markdown("---")
    st.caption("Nassau Candy Distributor · Decision Intelligence System")

# ──────────────────────────────────────────────
# LOAD DATA + FILTER
# ──────────────────────────────────────────────
df_raw = load_data()
df = df_raw.copy()
if sel_regions:   df = df[df["Region"].isin(sel_regions)]
if sel_factories: df = df[df["Factory"].isin(sel_factories)]
if sel_divisions: df = df[df["Division"].isin(sel_divisions)]
if sel_modes:     df = df[df["Ship Mode"].isin(sel_modes)]

# Train models on full dataset
model_results, le, feat_cols, best_model_name = train_models(df_raw)
best_model = model_results[best_model_name]["model"]

# ──────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────
st.markdown("# 🍬 Factory Reallocation & Shipping Optimization")
st.markdown("**Nassau Candy Distributor** · Decision Intelligence System · Predicts outcomes · Recommends reassignments · Balances efficiency and profitability")
st.markdown("---")

# ──────────────────────────────────────────────
# TABS
# ──────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview & EDA",
    "🤖 Predictive Model",
    "🏭 Factory Simulator",
    "🔀 What-If Analysis",
    "🏆 Recommendations",
])


# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW & EDA
# ══════════════════════════════════════════════
with tab1:
    # KPI row
    total_sales  = df["Sales"].sum()
    total_gp     = df["Gross Profit"].sum()
    avg_lead     = df["Lead Time"].mean()
    avg_margin   = df["Profit Margin (%)"].mean()
    total_orders = len(df)
    total_units  = df["Units"].sum()

    k1, k2, k3, k4, k5, k6 = st.columns(6)
    for col, label, value, delta, good in [
        (k1, "Total Sales",    f"${total_sales:,.0f}",   "",        True),
        (k2, "Gross Profit",   f"${total_gp:,.0f}",      "",        True),
        (k3, "Avg Lead Time",  f"{avg_lead:.1f} days",   "",        False),
        (k4, "Profit Margin",  f"{avg_margin:.1f}%",     "",        True),
        (k5, "Total Orders",   f"{total_orders:,}",      "",        True),
        (k6, "Total Units",    f"{total_units:,}",       "",        True),
    ]:
        col.markdown(f"""
        <div class="kpi-card">
            <p class="kpi-label">{label}</p>
            <p class="kpi-value">{value}</p>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📈 Sales & Profitability Analysis")
    c1, c2 = st.columns(2)

    with c1:
        # Sales by factory
        fac_sum = df.groupby("Factory")[["Sales", "Gross Profit", "Cost"]].sum().reset_index()
        fig = px.bar(fac_sum, x="Factory", y=["Sales", "Gross Profit", "Cost"],
                     barmode="group", title="Sales, Profit & Cost by Factory",
                     color_discrete_map={"Sales":"#4f46e5","Gross Profit":"#10b981","Cost":"#ef4444"},
                     labels={"value":"Amount ($)","variable":"Metric"})
        fig.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                          legend=dict(orientation="h", y=-0.2), height=370)
        fig.update_xaxes(tickangle=-20)
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        # Profit margin by division donut
        div_gp = df.groupby("Division")["Gross Profit"].sum().reset_index()
        fig2 = px.pie(div_gp, names="Division", values="Gross Profit",
                      title="Gross Profit Share by Division",
                      color="Division", color_discrete_map=DIV_COLORS, hole=0.45)
        fig2.update_layout(height=370, paper_bgcolor="white")
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("### 🚚 Shipping & Lead Time Analysis")
    c3, c4 = st.columns(2)

    with c3:
        # Average lead time by ship mode & region
        lt_pivot = df.groupby(["Ship Mode", "Region"])["Lead Time"].mean().reset_index()
        fig3 = px.bar(lt_pivot, x="Ship Mode", y="Lead Time", color="Region",
                      barmode="group", title="Avg Lead Time by Ship Mode & Region",
                      color_discrete_map=REGION_COLORS)
        fig3.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=370,
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig3, use_container_width=True)

    with c4:
        # Lead time distribution by factory — box plot
        fig4 = px.box(df, x="Factory", y="Lead Time", color="Factory",
                      title="Lead Time Distribution by Factory",
                      color_discrete_map=FACTORY_COLORS)
        fig4.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=370, showlegend=False)
        fig4.update_xaxes(tickangle=-20)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### 📅 Temporal Trends")
    c5, c6 = st.columns(2)

    with c5:
        # Monthly sales trend
        monthly = df.groupby(["Year", "Month", "Month Name"])["Sales"].sum().reset_index()
        monthly["Period"] = monthly["Year"].astype(str) + "-" + monthly["Month"].astype(str).str.zfill(2)
        monthly = monthly.sort_values("Period")
        fig5 = px.line(monthly, x="Period", y="Sales", title="Monthly Sales Trend",
                       markers=True, color_discrete_sequence=["#4f46e5"])
        fig5.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=340)
        fig5.update_xaxes(tickangle=-45, nticks=14)
        st.plotly_chart(fig5, use_container_width=True)

    with c6:
        # Ship mode distribution
        mode_cnt = df["Ship Mode"].value_counts().reset_index()
        mode_cnt.columns = ["Ship Mode", "Count"]
        fig6 = px.bar(mode_cnt, x="Ship Mode", y="Count",
                      title="Order Count by Ship Mode",
                      color="Ship Mode",
                      color_discrete_sequence=["#4f46e5","#10b981","#f59e0b","#ef4444"])
        fig6.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                           height=340, showlegend=False)
        st.plotly_chart(fig6, use_container_width=True)

    st.markdown("### 🗺️ Factory Map")
    map_df = pd.DataFrame([
        {"Factory": f, "lat": v["lat"], "lon": v["lon"],
         "Location": v["location"],
         "Total Sales": df_raw[df_raw["Factory"]==f]["Sales"].sum(),
         "Avg Lead Time": round(df_raw[df_raw["Factory"]==f]["Lead Time"].mean(), 1),
         "Orders": (df_raw["Factory"]==f).sum()}
        for f, v in FACTORY_COORDS.items()
    ])
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon", hover_name="Factory",
        hover_data={"Location": True, "Total Sales": ":$,.0f",
                    "Avg Lead Time": True, "Orders": True, "lat": False, "lon": False},
        size="Total Sales", color="Factory",
        color_discrete_map=FACTORY_COLORS,
        size_max=45, zoom=3.2,
        mapbox_style="carto-positron",
        title="Factory Locations (bubble = total sales)",
    )
    fig_map.update_layout(height=450, paper_bgcolor="white", margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig_map, use_container_width=True)

    # Raw data
    with st.expander("🔍 View Raw Data"):
        st.dataframe(df.reset_index(drop=True), use_container_width=True, height=350)
        csv_bytes = df.to_csv(index=False).encode()
        st.download_button("⬇️ Download Filtered CSV", csv_bytes,
                           "nassau_candy_filtered.csv", "text/csv")


# ══════════════════════════════════════════════
# TAB 2 — PREDICTIVE MODEL
# ══════════════════════════════════════════════
with tab2:
    st.markdown("## 🤖 Predictive Modeling — Lead Time Prediction")
    st.info("Three regression models trained on **Ship Mode, Region, Factory, Division** to predict shipping lead time. "
            "Best model selected by lowest RMSE.")

    # Model comparison metrics
    st.markdown("### Model Evaluation Results")
    mc1, mc2, mc3 = st.columns(3)
    for col, (name, res) in zip([mc1, mc2, mc3], model_results.items()):
        is_best = name == best_model_name
        card_cls = "model-metric best-model" if is_best else "model-metric"
        star = " ⭐ Best" if is_best else ""
        col.markdown(f"""
        <div class="{card_cls}">
            <div class="name">{name}{star}</div>
            <div class="stats">
                RMSE : <b>{res['rmse']}</b><br>
                MAE  : <b>{res['mae']}</b><br>
                R²   : <b>{res['r2']}</b>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("### 📊 Model Performance Charts")
    c1, c2 = st.columns(2)

    with c1:
        # RMSE comparison bar
        names = list(model_results.keys())
        rmses = [model_results[n]["rmse"] for n in names]
        maes  = [model_results[n]["mae"]  for n in names]
        r2s   = [model_results[n]["r2"]   for n in names]
        fig_m = go.Figure()
        fig_m.add_trace(go.Bar(name="RMSE", x=names, y=rmses, marker_color="#4f46e5"))
        fig_m.add_trace(go.Bar(name="MAE",  x=names, y=maes,  marker_color="#10b981"))
        fig_m.update_layout(title="RMSE & MAE Comparison", barmode="group",
                            plot_bgcolor="white", paper_bgcolor="white", height=340,
                            legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_m, use_container_width=True)

    with c2:
        # R² comparison
        fig_r2 = px.bar(x=names, y=r2s, title="R² Score Comparison",
                        color=names, color_discrete_sequence=["#4f46e5","#10b981","#f59e0b"],
                        labels={"x":"Model","y":"R² Score"})
        fig_r2.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                             height=340, showlegend=False)
        fig_r2.add_hline(y=1.0, line_dash="dot", line_color="gray", annotation_text="Perfect")
        st.plotly_chart(fig_r2, use_container_width=True)

    # Actual vs Predicted for best model
    st.markdown(f"### Actual vs Predicted Lead Time — {best_model_name}")
    y_test  = model_results[best_model_name]["y_test"]
    y_pred  = model_results[best_model_name]["preds"]
    scatter_df = pd.DataFrame({"Actual": y_test, "Predicted": np.round(y_pred, 2)})
    fig_scatter = px.scatter(scatter_df, x="Actual", y="Predicted",
                             opacity=0.6, color_discrete_sequence=["#4f46e5"],
                             title=f"{best_model_name} — Actual vs Predicted")
    max_val = max(scatter_df["Actual"].max(), scatter_df["Predicted"].max()) + 0.5
    fig_scatter.add_shape(type="line", x0=0, y0=0, x1=max_val, y1=max_val,
                          line=dict(color="#ef4444", dash="dash"))
    fig_scatter.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=420)
    st.plotly_chart(fig_scatter, use_container_width=True)

    # Feature importance (Random Forest)
    if "Random Forest" in model_results:
        st.markdown("### 🌳 Feature Importance (Random Forest)")
        rf_model = model_results["Random Forest"]["model"]
        imp = pd.DataFrame({"Feature": feat_cols, "Importance": rf_model.feature_importances_})
        imp = imp.sort_values("Importance", ascending=True)
        fig_imp = px.bar(imp, x="Importance", y="Feature", orientation="h",
                         title="Feature Importance for Lead Time Prediction",
                         color="Importance", color_continuous_scale="Blues")
        fig_imp.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                              height=300, coloraxis_showscale=False)
        st.plotly_chart(fig_imp, use_container_width=True)

    st.markdown("### 🔮 Single Prediction")
    p1, p2, p3, p4 = st.columns(4)
    sp_mode    = p1.selectbox("Ship Mode",  list(SHIP_BASE_LEAD.keys()), key="sp_mode")
    sp_region  = p2.selectbox("Region",     REGIONS,    key="sp_region")
    sp_factory = p3.selectbox("Factory",    FACTORIES,  key="sp_factory")
    sp_div     = p4.selectbox("Division",   ["Chocolate","Sugar","Other"], key="sp_div")

    pred_lead = predict_lead(best_model, le, feat_cols, sp_mode, sp_region, sp_factory, sp_div)
    base_lead  = SHIP_BASE_LEAD[sp_mode] + FACTORY_REGION_DIST[sp_factory][sp_region]
    delta_pct  = (pred_lead - base_lead) / base_lead * 100

    r1, r2, r3 = st.columns(3)
    r1.metric("🤖 Predicted Lead Time",    f"{pred_lead} days")
    r2.metric("📐 Rule-Based Lead Time",   f"{base_lead} days")
    r3.metric("Δ Model vs Rule",           f"{delta_pct:+.1f}%")


# ══════════════════════════════════════════════
# TAB 3 — FACTORY SIMULATOR
# ══════════════════════════════════════════════
with tab3:
    st.markdown("## 🏭 Factory Optimization Simulator")
    st.markdown("Select a product, region and ship mode to **compare predicted performance across all 5 factories**.")

    products_list = sorted(df_raw["Product Name"].unique())
    s1, s2, s3 = st.columns(3)
    sim_product = s1.selectbox("Product",   products_list, key="sim_prod")
    sim_region  = s2.selectbox("Region",    REGIONS,       key="sim_reg")
    sim_mode    = s3.selectbox("Ship Mode", list(SHIP_BASE_LEAD.keys()), key="sim_mode")

    sim_div = df_raw[df_raw["Product Name"] == sim_product]["Division"].iloc[0]
    cur_factory = PRODUCT_FACTORY.get(sim_product, FACTORIES[0])

    # Predict for all factories
    rows = []
    for fac in FACTORIES:
        pred_lt = predict_lead(best_model, le, feat_cols, sim_mode, sim_region, fac, sim_div)
        rule_lt = SHIP_BASE_LEAD[sim_mode] + FACTORY_REGION_DIST[fac][sim_region]
        prod_df = df_raw[(df_raw["Product Name"] == sim_product) & (df_raw["Factory"] == fac)]
        avg_gp  = prod_df["Gross Profit"].mean() if len(prod_df) else df_raw[df_raw["Product Name"] == sim_product]["Gross Profit"].mean()
        avg_sale= prod_df["Sales"].mean() if len(prod_df) else df_raw[df_raw["Product Name"] == sim_product]["Sales"].mean()
        orders  = len(prod_df)
        is_cur  = fac == cur_factory
        rows.append({
            "Factory": fac,
            "Is Current": is_cur,
            "Predicted Lead (days)": pred_lt,
            "Rule-Based Lead (days)": rule_lt,
            "Avg Gross Profit ($)": round(avg_gp, 2) if not np.isnan(avg_gp) else 0,
            "Avg Sales ($)": round(avg_sale, 2) if not np.isnan(avg_sale) else 0,
            "Historical Orders": orders,
        })

    sim_df = pd.DataFrame(rows).sort_values("Predicted Lead (days)")
    best_fac = sim_df.iloc[0]["Factory"]

    # KPIs
    cur_row  = sim_df[sim_df["Is Current"]].iloc[0]
    best_row = sim_df.iloc[0]
    lead_save = round(cur_row["Predicted Lead (days)"] - best_row["Predicted Lead (days)"], 2)
    gp_change = round(best_row["Avg Gross Profit ($)"] - cur_row["Avg Gross Profit ($)"], 2)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Current Factory",            cur_factory)
    m2.metric("Recommended Factory",        best_fac,
              delta=None if best_fac == cur_factory else "✅ Switch recommended")
    m3.metric("Lead Time Saving",           f"{lead_save} days",
              delta=f"{lead_save} days faster" if lead_save > 0 else "Already optimal")
    m4.metric("GP Impact",                  f"${gp_change:+.2f}",
              delta_color="normal")

    st.markdown("### Factory Performance Comparison")
    c1, c2 = st.columns(2)

    with c1:
        colors = [FACTORY_COLORS.get(f, "#888") for f in sim_df["Factory"]]
        fig_sim = go.Figure()
        fig_sim.add_trace(go.Bar(
            x=sim_df["Factory"], y=sim_df["Predicted Lead (days)"],
            marker_color=colors, name="Predicted Lead",
            text=sim_df["Predicted Lead (days)"].apply(lambda x: f"{x}d"),
            textposition="outside",
        ))
        fig_sim.add_hline(y=cur_row["Predicted Lead (days)"], line_dash="dash",
                          line_color="#ef4444", annotation_text=f"Current ({cur_factory})")
        fig_sim.update_layout(title="Predicted Lead Time by Factory",
                              plot_bgcolor="white", paper_bgcolor="white",
                              height=380, yaxis_title="Lead Time (days)")
        fig_sim.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_sim, use_container_width=True)

    with c2:
        fig_gp = go.Figure()
        fig_gp.add_trace(go.Bar(
            x=sim_df["Factory"], y=sim_df["Avg Gross Profit ($)"],
            marker_color=colors, name="Avg Gross Profit",
            text=sim_df["Avg Gross Profit ($)"].apply(lambda x: f"${x:.2f}"),
            textposition="outside",
        ))
        fig_gp.update_layout(title="Avg Gross Profit per Order by Factory",
                             plot_bgcolor="white", paper_bgcolor="white",
                             height=380, yaxis_title="Avg Gross Profit ($)")
        fig_gp.update_xaxes(tickangle=-20)
        st.plotly_chart(fig_gp, use_container_width=True)

    # Table
    st.markdown("### Detailed Factory Scorecard")
    display_sim = sim_df.copy()
    display_sim.insert(0, "Rank", range(1, len(display_sim)+1))
    display_sim["Status"] = display_sim.apply(
        lambda r: "✅ Recommended" if r["Factory"] == best_fac and not r["Is Current"]
                  else ("📍 Current" if r["Is Current"] else "—"), axis=1
    )
    st.dataframe(
        display_sim[["Rank","Factory","Status","Predicted Lead (days)",
                     "Rule-Based Lead (days)","Avg Gross Profit ($)","Avg Sales ($)","Historical Orders"]],
        use_container_width=True, hide_index=True
    )

    # Radar chart
    st.markdown("### 🕸️ Multi-Metric Factory Radar")
    cats = ["Speed Score", "Profit Score", "Order Volume Score", "Efficiency"]
    max_lt  = sim_df["Predicted Lead (days)"].max()
    max_gp  = sim_df["Avg Gross Profit ($)"].max() or 1
    max_ord = sim_df["Historical Orders"].max() or 1
    fig_radar = go.Figure()
    for _, row in sim_df.iterrows():
        speed   = round((max_lt - row["Predicted Lead (days)"]) / max_lt * 10, 2)
        profit  = round(row["Avg Gross Profit ($)"] / max_gp * 10, 2)
        vol     = round(row["Historical Orders"] / max_ord * 10, 2)
        eff     = round((speed + profit) / 2, 2)
        fig_radar.add_trace(go.Scatterpolar(
            r=[speed, profit, vol, eff, speed],
            theta=cats + [cats[0]],
            name=row["Factory"],
            line_color=FACTORY_COLORS.get(row["Factory"], "#888"),
            fill="toself", opacity=0.3,
        ))
    fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0,10])),
                            height=430, paper_bgcolor="white",
                            legend=dict(orientation="h", y=-0.1))
    st.plotly_chart(fig_radar, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 4 — WHAT-IF ANALYSIS
# ══════════════════════════════════════════════
with tab4:
    st.markdown("## 🔀 What-If Scenario Analysis")
    st.markdown("Compare **current vs recommended** factory assignments. Adjust filters to explore scenarios.")

    wc1, wc2 = st.columns(2)
    wi_region = wc1.selectbox("Target Region", ["All"] + REGIONS, key="wi_reg")
    wi_mode   = wc2.selectbox("Ship Mode Filter", ["All"] + list(SHIP_BASE_LEAD.keys()), key="wi_mode")

    wdf = df_raw.copy()
    if wi_region != "All": wdf = wdf[wdf["Region"] == wi_region]
    if wi_mode   != "All": wdf = wdf[wdf["Ship Mode"] == wi_mode]

    # Current avg lead by factory
    cur_leads  = wdf.groupby("Factory")["Lead Time"].mean().round(2)

    # Recommended lead — best alternate factory per product
    speed_w  = priority / 100
    profit_w = 1 - speed_w

    # Build product recommendations
    prod_recs = []
    for prod in df_raw["Product Name"].unique():
        prod_rows = df_raw[df_raw["Product Name"] == prod]
        cur_fac   = PRODUCT_FACTORY.get(prod, FACTORIES[0])
        sim_div   = prod_rows["Division"].iloc[0]
        avg_gp    = prod_rows["Gross Profit"].mean()

        best_score = -np.inf
        best_fac_r = cur_fac
        scores = {}
        for fac in FACTORIES:
            tgt_region = wi_region if wi_region != "All" else "Interior"
            tgt_mode   = wi_mode   if wi_mode   != "All" else "Standard Class"
            pred_lt = predict_lead(best_model, le, feat_cols, tgt_mode, tgt_region, fac, sim_div)
            fac_gp  = df_raw[df_raw["Factory"]==fac]["Gross Profit"].mean()
            # Normalize
            speed_score  = (10 - pred_lt) / 10
            profit_score = fac_gp / (df_raw["Gross Profit"].max() or 1)
            score = speed_w * speed_score + profit_w * profit_score
            scores[fac] = {"lead": pred_lt, "gp": fac_gp, "score": score}
            if score > best_score and fac != cur_fac:
                best_score = score
                best_fac_r = fac

        cur_lt  = scores[cur_fac]["lead"]
        best_lt = scores[best_fac_r]["lead"]
        delta_lt = round(cur_lt - best_lt, 2)
        gp_cur  = scores[cur_fac]["gp"]
        gp_new  = scores[best_fac_r]["gp"]
        gp_chg  = round((gp_new - gp_cur), 2)
        conf    = round(min(99, 55 + abs(delta_lt)*8 + speed_w*15), 0)
        risk    = "low" if abs(gp_chg) < 1 and delta_lt >= 0 else ("medium" if abs(gp_chg) < 3 else "high")

        prod_recs.append({
            "Product": prod,
            "Division": sim_div,
            "Current Factory": cur_fac,
            "Recommended Factory": best_fac_r,
            "Current Lead (days)": cur_lt,
            "Recommended Lead (days)": best_lt,
            "Lead Δ (days)": -delta_lt,
            "GP Impact ($)": gp_chg,
            "Confidence (%)": conf,
            "Risk": risk,
            "Change": best_fac_r != cur_fac,
        })

    rec_df = pd.DataFrame(prod_recs)

    # Current vs Recommended lead by factory
    st.markdown("### 📊 Current vs Recommended Lead Time by Factory")
    target_region_for_chart = wi_region if wi_region != "All" else "Interior"
    target_mode_for_chart   = wi_mode   if wi_mode   != "All" else "Standard Class"
    sim_div_chart = "Chocolate"
    rec_leads_map = {}
    for fac in FACTORIES:
        rec_leads_map[fac] = predict_lead(best_model, le, feat_cols,
                                          target_mode_for_chart, target_region_for_chart, fac, sim_div_chart)

    fig_wi = go.Figure()
    fig_wi.add_trace(go.Bar(
        x=FACTORIES, y=[cur_leads.get(f, 0) for f in FACTORIES],
        name="Current Avg Lead", marker_color="#94a3b8",
        text=[f"{cur_leads.get(f,0):.1f}d" for f in FACTORIES], textposition="outside",
    ))
    fig_wi.add_trace(go.Bar(
        x=FACTORIES, y=[rec_leads_map[f] for f in FACTORIES],
        name="Predicted Optimal Lead", marker_color="#4f46e5",
        text=[f"{rec_leads_map[f]:.1f}d" for f in FACTORIES], textposition="outside",
    ))
    fig_wi.update_layout(barmode="group", plot_bgcolor="white", paper_bgcolor="white",
                         height=390, yaxis_title="Lead Time (days)",
                         legend=dict(orientation="h", y=-0.2))
    fig_wi.update_xaxes(tickangle=-20)
    st.plotly_chart(fig_wi, use_container_width=True)

    # Efficiency gains
    st.markdown("### 📈 Projected Efficiency Gains")
    g1, g2, g3, g4 = st.columns(4)
    recs_with_change  = rec_df[rec_df["Change"]]
    avg_lead_reduce   = recs_with_change["Lead Δ (days)"].mean() if len(recs_with_change) else 0
    total_gp_impact   = recs_with_change["GP Impact ($)"].sum()
    avg_conf          = rec_df["Confidence (%)"].mean()
    low_risk_count    = (rec_df["Risk"] == "low").sum()

    g1.metric("Products to Reassign",  len(recs_with_change))
    g2.metric("Avg Lead Reduction",    f"{avg_lead_reduce:.1f} days")
    g3.metric("Total GP Impact",       f"${total_gp_impact:+.2f}")
    g4.metric("Avg Confidence",        f"{avg_conf:.0f}%")

    c1, c2 = st.columns(2)
    with c1:
        # Lead improvement per product
        change_df = rec_df[rec_df["Change"]].sort_values("Lead Δ (days)")
        if len(change_df):
            fig_ld = px.bar(change_df, x="Lead Δ (days)", y="Product",
                            orientation="h", title="Lead Time Reduction per Product",
                            color="Lead Δ (days)", color_continuous_scale="Blues",
                            labels={"Lead Δ (days)": "Days Saved"})
            fig_ld.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                                 height=400, coloraxis_showscale=False)
            st.plotly_chart(fig_ld, use_container_width=True)

    with c2:
        # GP impact
        if len(change_df):
            colors_gp = ["#10b981" if v >= 0 else "#ef4444" for v in change_df["GP Impact ($)"]]
            fig_gp2 = go.Figure(go.Bar(
                x=change_df["Product"], y=change_df["GP Impact ($)"],
                marker_color=colors_gp,
                text=change_df["GP Impact ($)"].apply(lambda x: f"${x:+.2f}"),
                textposition="outside",
            ))
            fig_gp2.update_layout(title="Gross Profit Impact per Product",
                                  plot_bgcolor="white", paper_bgcolor="white",
                                  height=400, yaxis_title="GP Change ($)")
            fig_gp2.update_xaxes(tickangle=-45)
            st.plotly_chart(fig_gp2, use_container_width=True)

    # Scenario summary metrics
    st.markdown("### 🎚️ Optimization Priority Analysis")
    priority_range = list(range(0, 101, 10))
    scenario_rows = []
    for p in priority_range:
        sw = p / 100
        pw = 1 - sw
        total_saving = 0
        total_gp_imp = 0
        for prod in df_raw["Product Name"].unique():
            cur_fac  = PRODUCT_FACTORY.get(prod, FACTORIES[0])
            sim_div2 = df_raw[df_raw["Product Name"]==prod]["Division"].iloc[0]
            best_sc  = -np.inf
            best_lt_p = predict_lead(best_model, le, feat_cols,
                                    target_mode_for_chart, target_region_for_chart, cur_fac, sim_div2)
            best_gp_p = df_raw[df_raw["Factory"]==cur_fac]["Gross Profit"].mean()
            for fac in FACTORIES:
                if fac == cur_fac: continue
                lt_p = predict_lead(best_model, le, feat_cols,
                                   target_mode_for_chart, target_region_for_chart, fac, sim_div2)
                gp_p = df_raw[df_raw["Factory"]==fac]["Gross Profit"].mean()
                sc   = sw*((10-lt_p)/10) + pw*(gp_p/(df_raw["Gross Profit"].max() or 1))
                if sc > best_sc:
                    best_sc = sc
                    best_lt_p = lt_p
                    best_gp_p = gp_p
            cur_lt2 = predict_lead(best_model, le, feat_cols,
                                  target_mode_for_chart, target_region_for_chart, cur_fac, sim_div2)
            cur_gp2 = df_raw[df_raw["Factory"]==cur_fac]["Gross Profit"].mean()
            total_saving += (cur_lt2 - best_lt_p)
            total_gp_imp += (best_gp_p - cur_gp2)
        scenario_rows.append({"Priority (Speed %)": p,
                               "Total Lead Saving (days)": round(total_saving, 1),
                               "Total GP Impact ($)": round(total_gp_imp, 2)})

    sc_df = pd.DataFrame(scenario_rows)
    fig_sc = make_subplots(specs=[[{"secondary_y": True}]])
    fig_sc.add_trace(go.Scatter(x=sc_df["Priority (Speed %)"], y=sc_df["Total Lead Saving (days)"],
                                 name="Lead Saving (days)", mode="lines+markers",
                                 line=dict(color="#4f46e5", width=2)), secondary_y=False)
    fig_sc.add_trace(go.Scatter(x=sc_df["Priority (Speed %)"], y=sc_df["Total GP Impact ($)"],
                                 name="GP Impact ($)", mode="lines+markers",
                                 line=dict(color="#10b981", width=2, dash="dash")), secondary_y=True)
    fig_sc.add_vline(x=priority, line_dash="dot", line_color="#ef4444",
                     annotation_text=f"Current: {priority}%")
    fig_sc.update_xaxes(title_text="Speed Priority (%)")
    fig_sc.update_yaxes(title_text="Total Lead Saving (days)", secondary_y=False)
    fig_sc.update_yaxes(title_text="Total GP Impact ($)", secondary_y=True)
    fig_sc.update_layout(title="Speed vs Profit Trade-off Across Priority Settings",
                         plot_bgcolor="white", paper_bgcolor="white", height=390,
                         legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(fig_sc, use_container_width=True)


# ══════════════════════════════════════════════
# TAB 5 — RECOMMENDATIONS
# ══════════════════════════════════════════════
with tab5:
    st.markdown("## 🏆 Factory Reassignment Recommendations")
    st.markdown("Ranked by combined **lead time reduction + profit impact** score based on your priority setting.")

    # Sort by composite score
    rec_df["Score"] = (
        (priority / 100) * rec_df["Lead Δ (days)"].clip(lower=0) / (rec_df["Lead Δ (days)"].max() or 1) * 10 +
        (1 - priority / 100) * rec_df["GP Impact ($)"] / (rec_df["GP Impact ($)"].abs().max() or 1) * 10
    )
    top_recs = rec_df[rec_df["Change"]].sort_values("Score", ascending=False).head(top_n).reset_index(drop=True)

    # Summary KPIs
    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("Recommendations Generated",  len(top_recs))
    rk2.metric("Low Risk",   (top_recs["Risk"]=="low").sum(),   delta="✅ Safe to proceed")
    rk3.metric("Medium Risk",(top_recs["Risk"]=="medium").sum(),delta_color="off")
    rk4.metric("High Risk",  (top_recs["Risk"]=="high").sum(),  delta="⚠️ Review needed", delta_color="inverse")

    # Recommendation table
    st.markdown("### 📋 Top Reassignment Recommendations")
    rows_html = ""
    for i, row in top_recs.iterrows():
        lead_delta_str = f"−{abs(row['Lead Δ (days)']):.1f}d" if row['Lead Δ (days)'] < 0 else f"+{row['Lead Δ (days)']:.1f}d"
        gp_color = "#10b981" if row["GP Impact ($)"] >= 0 else "#ef4444"
        rows_html += f"""
        <tr>
            <td><b>{i+1}</b></td>
            <td>{row['Product']}</td>
            <td><span class="badge badge-cur">{row['Current Factory']}</span></td>
            <td><span class="badge badge-rec">{row['Recommended Factory']}</span></td>
            <td style="color:#4f46e5;font-weight:600">{row['Current Lead (days)']:.1f}d → {row['Recommended Lead (days)']:.1f}d</td>
            <td style="color:#10b981;font-weight:600">−{abs(row['Lead Δ (days)']):.1f}d</td>
            <td style="color:{gp_color};font-weight:600">${row['GP Impact ($)']:+.2f}</td>
            <td>{row['Confidence (%)']:.0f}%</td>
            <td>{badge(row['Risk'])}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto">
    <table class="rec-table">
      <thead>
        <tr>
          <th>#</th><th>Product</th><th>Current Factory</th><th>Recommended Factory</th>
          <th>Lead Time</th><th>Lead Δ</th><th>GP Impact</th><th>Confidence</th><th>Risk</th>
        </tr>
      </thead>
      <tbody>{rows_html}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### 📊 Recommendation Charts")
    ch1, ch2 = st.columns(2)

    with ch1:
        # Confidence score per recommendation
        fig_conf = px.bar(top_recs, x="Confidence (%)", y="Product",
                          orientation="h", color="Confidence (%)",
                          color_continuous_scale="Purples",
                          title="Confidence Score by Product",
                          labels={"Confidence (%)": "Confidence (%)"})
        fig_conf.update_layout(plot_bgcolor="white", paper_bgcolor="white",
                               height=max(350, len(top_recs)*40 + 60),
                               coloraxis_showscale=False)
        st.plotly_chart(fig_conf, use_container_width=True)

    with ch2:
        # Risk breakdown pie
        risk_counts = top_recs["Risk"].value_counts().reset_index()
        risk_counts.columns = ["Risk", "Count"]
        fig_risk = px.pie(risk_counts, names="Risk", values="Count",
                          color="Risk", title="Risk Distribution",
                          color_discrete_map={"low":"#10b981","medium":"#f59e0b","high":"#ef4444"},
                          hole=0.4)
        fig_risk.update_layout(height=350, paper_bgcolor="white")
        st.plotly_chart(fig_risk, use_container_width=True)

    # Risk & Impact panel
    st.markdown("### ⚠️ Risk & Impact Panel")
    rp1, rp2, rp3 = st.columns(3)
    for col, level, color, icon in [
        (rp1, "low",    "#d1fae5", "✅"),
        (rp2, "medium", "#fef3c7", "⚠️"),
        (rp3, "high",   "#fee2e2", "🚨"),
    ]:
        items = top_recs[top_recs["Risk"] == level]
        products_list_html = "".join(
            f"<li style='font-size:13px;margin-bottom:4px'>{r['Product']} "
            f"<span style='color:#4f46e5'>→ {r['Recommended Factory']}</span></li>"
            for _, r in items.iterrows()
        ) if len(items) else "<li style='font-size:13px;color:#94a3b8'>No items</li>"
        col.markdown(f"""
        <div style="background:{color};border-radius:10px;padding:14px 16px;min-height:180px">
            <b style="font-size:14px">{icon} {level.upper()} Risk — {len(items)} item(s)</b>
            <ul style="margin-top:8px;padding-left:16px">{products_list_html}</ul>
        </div>""", unsafe_allow_html=True)

    # Scenario confidence scatter
    st.markdown("### 🎯 Lead Reduction vs GP Impact (Scenario Confidence Map)")
    fig_bubble = px.scatter(
        top_recs,
        x="Lead Δ (days)", y="GP Impact ($)",
        size="Confidence (%)", color="Risk",
        hover_name="Product",
        hover_data={"Current Factory": True, "Recommended Factory": True,
                    "Confidence (%)": True},
        color_discrete_map={"low":"#10b981","medium":"#f59e0b","high":"#ef4444"},
        title="Lead Reduction vs GP Impact — Bubble Size = Confidence",
        size_max=45,
    )
    fig_bubble.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
    fig_bubble.add_vline(x=0, line_dash="dash", line_color="#94a3b8")
    fig_bubble.update_layout(plot_bgcolor="white", paper_bgcolor="white", height=430)
    st.plotly_chart(fig_bubble, use_container_width=True)

    # Factory reallocation flow (Sankey)
    st.markdown("### 🔄 Reallocation Flow (Sankey Diagram)")
    sankey_df = top_recs.groupby(["Current Factory","Recommended Factory"]).size().reset_index(name="Count")
    all_nodes  = list(set(sankey_df["Current Factory"].tolist() + sankey_df["Recommended Factory"].tolist()))
    node_idx   = {n: i for i, n in enumerate(all_nodes)}
    node_colors= [FACTORY_COLORS.get(n,"#888") for n in all_nodes]
    fig_sankey = go.Figure(go.Sankey(
        node=dict(label=all_nodes, color=node_colors, pad=20, thickness=20),
        link=dict(
            source=[node_idx[r["Current Factory"]]     for _, r in sankey_df.iterrows()],
            target=[node_idx[r["Recommended Factory"]] for _, r in sankey_df.iterrows()],
            value= sankey_df["Count"].tolist(),
            color=["rgba(79,70,229,0.3)"] * len(sankey_df),
        ),
    ))
    fig_sankey.update_layout(title="Product Reallocation Flow",
                             height=380, paper_bgcolor="white", font_size=13)
    st.plotly_chart(fig_sankey, use_container_width=True)

    # Download recommendations
    st.markdown("### ⬇️ Export")
    ex1, ex2 = st.columns(2)
    csv_rec = top_recs.drop(columns=["Change","Score"], errors="ignore").to_csv(index=False).encode()
    ex1.download_button("📥 Download Recommendations CSV", csv_rec,
                        "nassau_recommendations.csv", "text/csv")
    full_csv = rec_df.drop(columns=["Change","Score"], errors="ignore").to_csv(index=False).encode()
    ex2.download_button("📥 Download Full Product Analysis CSV", full_csv,
                        "nassau_full_analysis.csv", "text/csv")

    # Executive Summary
    st.markdown("---")
    st.markdown("### 📝 Executive Summary")
    low_c  = (top_recs["Risk"]=="low").sum()
    med_c  = (top_recs["Risk"]=="medium").sum()
    high_c = (top_recs["Risk"]=="high").sum()
    best_lead_prod = top_recs.loc[top_recs["Lead Δ (days)"].idxmin(), "Product"] if len(top_recs) else "N/A"
    avg_conf_ex    = top_recs["Confidence (%)"].mean()
    st.markdown(f"""
    <div class="section-card">
    <p>This analysis processed <b>{len(df_raw):,} orders</b> across <b>5 factories</b>, <b>4 regions</b>,
    and <b>15 products</b>. The best-performing predictive model was <b>{best_model_name}</b>
    (RMSE={model_results[best_model_name]['rmse']}, R²={model_results[best_model_name]['r2']}).</p>

    <p>With an optimization priority of <b>{priority}% speed / {100-priority}% profit</b>,
    the system identified <b>{len(top_recs)} reassignment opportunities</b>:
    <span style="color:#059669"><b>{low_c} low risk</b></span>,
    <span style="color:#d97706"><b>{med_c} medium risk</b></span>,
    <span style="color:#dc2626"><b>{high_c} high risk</b></span>.</p>

    <p>The greatest lead time improvement is for <b>{best_lead_prod}</b>.
    Average model confidence across recommendations: <b>{avg_conf_ex:.0f}%</b>.</p>

    <p>These recommendations elevate Nassau Candy Distributor from descriptive analytics to
    <b>intelligent decision-making</b> — combining predictive modeling with optimization logic
    to improve shipping efficiency without sacrificing profitability.</p>
    </div>
    """, unsafe_allow_html=True)
