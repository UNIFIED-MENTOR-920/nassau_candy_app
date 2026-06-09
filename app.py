# ============================================================
#  Nassau Candy Distributor
#  Factory Reallocation & Shipping Optimization System
#  Streamlit Dashboard  |  deploy-ready
# ============================================================

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

# -------------------------------------------------------------
# PAGE CONFIG  (must be first Streamlit call)
# -------------------------------------------------------------
st.set_page_config(
    page_title="Nassau Candy | Factory Optimization",
    page_icon="??",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -------------------------------------------------------------
# CSS  � works for both Streamlit light AND dark themes
#        Uses CSS variables that Streamlit injects per-theme
# -------------------------------------------------------------
st.markdown("""
<style>
/* ----- KPI cards ----- */
.kpi-wrap {
    /* Use a subtle translucent background so cards work on both themes */
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 18px 20px 14px;
    margin-bottom: 4px;
    border-left: 4px solid var(--kpi-accent, #4f46e5);
}
.kpi-label {
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .8px;
    text-transform: uppercase;
    color: var(--text-color, #e6e6e6);
    opacity: .55;
    margin: 0 0 6px;
}
.kpi-value {
    font-size: 26px;
    font-weight: 700;
    color: var(--text-color, #fff);
    margin: 0;
    line-height: 1.1;
}
.kpi-sub {
    font-size: 11px;
    margin: 4px 0 0;
    opacity: .55;
    color: var(--text-color, #d1d5db);
}

/* ----- section separator ----- */
.sec-title {
    font-size: 15px;
    font-weight: 600;
    letter-spacing: .3px;
    margin: 24px 0 12px;
    padding-bottom: 6px;
    border-bottom: 1px solid rgba(128,128,128,.18);
    color: var(--text-color);
}

/* ----- recommendation table ----- */
.rtable { width:100%; border-collapse:collapse; font-size:13px; }
.rtable th {
    padding: 9px 12px;
    text-align: left;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .6px;
    text-transform: uppercase;
    border-bottom: 2px solid rgba(128,128,128,.2);
    color: var(--text-color);
    opacity: .6;
}
.rtable td {
    padding: 9px 12px;
    border-bottom: 1px solid rgba(128,128,128,.1);
    color: var(--text-color);
    vertical-align: middle;
}
.rtable tr:hover td { background: rgba(79,70,229,.06); }

/* ----- badges ----- */
.badge {
    display: inline-block;
    padding: 2px 9px;
    border-radius: 20px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: .3px;
}
.b-low    { background: #d1fae5; color: #065f46; }
.b-medium { background: #fef3c7; color: #92400e; }
.b-high   { background: #fee2e2; color: #991b1b; }
.b-cur    { background: #e0f2fe; color: #0369a1; }
.b-rec    { background: #ede9fe; color: #5b21b6; }

/* ----- model card ----- */
.mcard {
    border: 1px solid rgba(128,128,128,.2);
    border-radius: 10px;
    padding: 14px 16px;
    margin-bottom: 6px;
}
.mcard-best { border: 2px solid #4f46e5 !important; }
.mcard-name { font-size: 14px; font-weight: 700; margin-bottom: 6px; }
.mcard-stat { font-size: 12px; opacity: .7; line-height: 1.7; }

/* ----- risk panel ----- */
.rpanel {
    border-radius: 10px;
    padding: 14px 16px;
    min-height: 160px;
}
.rpanel-title { font-size: 13px; font-weight: 700; margin-bottom: 8px; }
.rpanel ul { margin: 0; padding-left: 14px; }
.rpanel li { font-size: 12px; margin-bottom: 3px; opacity: .85; }

/* ----- exec summary ----- */
.exec-box {
    border: 1px solid rgba(128,128,128,.18);
    border-radius: 12px;
    padding: 20px 24px;
    line-height: 1.75;
    font-size: 14px;
}

/* ----- sidebar ----- */
[data-testid="stSidebar"] { border-right: 1px solid rgba(128,128,128,.15); }
.sidebar-logo { font-size: 22px; font-weight: 800; letter-spacing: -.3px; margin-bottom: 2px; }
.sidebar-tagline { font-size: 11px; opacity: .55; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# CONSTANTS
# -------------------------------------------------------------
PRODUCT_FACTORY = {
    "Wonka Bar - Nutty Crunch Surprise":  "Lot's O' Nuts",
    "Wonka Bar - Fudge Mallows":          "Lot's O' Nuts",
    "Wonka Bar -Scrumdiddlyumptious":     "Lot's O' Nuts",
    "Wonka Bar - Scrumdiddlyumptious":    "Lot's O' Nuts",
    "Wonka Bar - Milk Chocolate":         "Wicked Choccy's",
    "Wonka Bar - Triple Dazzle Caramel":  "Wicked Choccy's",
    "Laffy Taffy":                        "Sugar Shack",
    "SweeTARTS":                          "Sugar Shack",
    "Nerds":                              "Sugar Shack",
    "Fun Dip":                            "Sugar Shack",
    "Fizzy Lifting Drinks":               "Sugar Shack",
    "Everlasting Gobstopper":             "Secret Factory",
    "Hair Toffee":                        "The Other Factory",
    "Lickable Wallpaper":                 "Secret Factory",
    "Wonka Gum":                          "Secret Factory",
    "Kazookles":                          "The Other Factory",
}

FACTORY_REGION_DIST = {
    "Lot's O' Nuts":     {"Interior": 1, "Atlantic": 2, "Gulf": 1, "Pacific": 0},
    "Wicked Choccy's":   {"Interior": 1, "Atlantic": 0, "Gulf": 1, "Pacific": 2},
    "Sugar Shack":       {"Interior": 0, "Atlantic": 1, "Gulf": 1, "Pacific": 2},
    "Secret Factory":    {"Interior": 0, "Atlantic": 1, "Gulf": 1, "Pacific": 2},
    "The Other Factory": {"Interior": 0, "Atlantic": 1, "Gulf": 0, "Pacific": 1},
}

FACTORY_COORDS = {
    "Lot's O' Nuts":     (32.881893, -111.768036),
    "Wicked Choccy's":   (32.076176,  -81.088371),
    "Sugar Shack":       (48.119140,  -96.181150),
    "Secret Factory":    (41.446333,  -90.565487),
    "The Other Factory": (35.117500,  -89.971107),
}

SHIP_BASE = {"Standard Class": 5, "Second Class": 3, "First Class": 2, "Same Day": 1}
FACTORIES  = list(FACTORY_COORDS.keys())
REGIONS    = ["Interior", "Atlantic", "Gulf", "Pacific"]
DIVISIONS  = ["Chocolate", "Sugar", "Other"]
SHIP_MODES = list(SHIP_BASE.keys())

F_COLOR = {
    "Lot's O' Nuts":     "#4f46e5",
    "Wicked Choccy's":   "#e11d48",
    "Sugar Shack":       "#059669",
    "Secret Factory":    "#d97706",
    "The Other Factory": "#7c3aed",
}
R_COLOR = {"Interior": "#4f46e5", "Atlantic": "#0ea5e9", "Gulf": "#10b981", "Pacific": "#f59e0b"}
D_COLOR = {"Chocolate": "#92400e", "Sugar": "#ec4899", "Other": "#6366f1"}
M_COLOR = {
    "Standard Class": "#6366f1",
    "Second Class":   "#0ea5e9",
    "First Class":    "#10b981",
    "Same Day":       "#f59e0b",
}

PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(showgrid=False, zeroline=False),
    yaxis=dict(gridcolor="rgba(128,128,128,.12)", zeroline=False),
    legend=dict(orientation="h", y=-0.22, x=0),
)

def pl(fig, h=360):
    fig.update_layout(**PLOTLY_LAYOUT, height=h)
    return fig

# -------------------------------------------------------------
# DATA LOADING
# -------------------------------------------------------------
@st.cache_data(show_spinner="Loading dataset ...")
def load_data():
    import os
    from pathlib import Path

    base_dir = Path(__file__).resolve().parent
    possible_paths = [
        base_dir / "data" / "orders.csv",
        base_dir / "Nassau_Candy_Distributor.csv",
        base_dir / "orders.csv",
    ]
    dataset_path = next((p for p in possible_paths if p.exists()), None)

    if dataset_path is None:
        st.error(
            "? Dataset not found. Place `orders.csv` in the `data/` folder or `Nassau_Candy_Distributor.csv` next to `app.py`.")
        st.stop()

    df = pd.read_csv(dataset_path)
    for c in ["Sales", "Cost", "Gross Profit", "Units"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True, errors="coerce")
    df["Ship Date"]  = pd.to_datetime(df["Ship Date"],  dayfirst=True, errors="coerce")

    df["Factory"]  = df["Product Name"].map(PRODUCT_FACTORY)
    df["Lead Time"] = df.apply(
        lambda r: SHIP_BASE.get(r["Ship Mode"], 5)
                  + FACTORY_REGION_DIST.get(r["Factory"], {}).get(r["Region"], 1),
        axis=1,
    )
    df["Profit Margin (%)"] = (df["Gross Profit"] / df["Sales"] * 100).round(2)
    df["Year"]       = df["Order Date"].dt.year
    df["Month"]      = df["Order Date"].dt.month
    df["Month Label"] = df["Order Date"].dt.strftime("%Y-%b")
    df.dropna(subset=["Factory", "Sales", "Lead Time"], inplace=True)
    return df

# -------------------------------------------------------------
# MODEL TRAINING
# -------------------------------------------------------------
@st.cache_resource(show_spinner="Training models �")
def train_models(df_hash):
    df = load_data()
    feats  = ["Ship Mode", "Region", "Factory", "Division"]
    target = "Lead Time"
    sub    = df[feats + [target]].dropna()
    le = {}
    X  = pd.DataFrame()
    for c in feats:
        enc    = LabelEncoder()
        X[c]   = enc.fit_transform(sub[c].astype(str))
        le[c]  = enc
    y = sub[target]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42)

    specs = {
        "Linear Regression":  LinearRegression(),
        "Random Forest":      RandomForestRegressor(n_estimators=150, random_state=42, n_jobs=-1),
        "Gradient Boosting":  GradientBoostingRegressor(n_estimators=150, random_state=42),
    }
    results = {}
    for name, mdl in specs.items():
        mdl.fit(X_tr, y_tr)
        p = mdl.predict(X_te)
        results[name] = dict(
            model=mdl,
            rmse=round(float(np.sqrt(mean_squared_error(y_te, p))), 4),
            mae= round(float(mean_absolute_error(y_te, p)), 4),
            r2=  round(float(r2_score(y_te, p)), 4),
            preds=p, y_test=y_te.values,
        )
    best = min(results, key=lambda k: results[k]["rmse"])
    return results, le, feats, best


def pred_lead(model, le, feats, mode, region, factory, division):
    row = {"Ship Mode": mode, "Region": region, "Factory": factory, "Division": division}
    try:
        X = np.array([[le[f].transform([row[f]])[0] for f in feats]])
        return round(float(model.predict(X)[0]), 2)
    except Exception:
        return SHIP_BASE.get(mode, 5) + FACTORY_REGION_DIST.get(factory, {}).get(region, 1)


def risk_badge(level):
    cls = {"low": "b-low", "medium": "b-medium", "high": "b-high"}.get(level, "b-low")
    return f'<span class="badge {cls}">{level.upper()}</span>'

# -------------------------------------------------------------
# SIDEBAR
# -------------------------------------------------------------
with st.sidebar:
    st.markdown('<div class="sidebar-logo">?? Nassau Candy</div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-tagline">Factory Reallocation & Shipping Optimization</div>',
                unsafe_allow_html=True)
    st.divider()

    st.markdown("**Global Filters**")
    sel_regions   = st.multiselect("Region",    REGIONS,    default=REGIONS)
    sel_factories = st.multiselect("Factory",   FACTORIES,  default=FACTORIES)
    sel_divisions = st.multiselect("Division",  DIVISIONS,  default=DIVISIONS)
    sel_modes     = st.multiselect("Ship Mode", SHIP_MODES, default=SHIP_MODES)

    st.divider()
    st.markdown("**Optimization Settings**")
    priority = st.slider(
        "Speed  ?------?  Profit",
        min_value=0, max_value=100, value=50, step=5,
        help="0 = maximise profit  |  100 = minimise lead time",
    )
    top_n = st.slider("Top-N Recommendations", 3, 15, 7)

    st.divider()
    st.caption("Nassau Candy Distributor  �  Decision Intelligence System")

# -------------------------------------------------------------
# LOAD + FILTER
# -------------------------------------------------------------
df_raw  = load_data()
df_hash = len(df_raw)          # stable cache key
model_results, le_map, feat_cols, best_name = train_models(df_hash)
best_mdl = model_results[best_name]["model"]

df = df_raw.copy()
if sel_regions:   df = df[df["Region"].isin(sel_regions)]
if sel_factories: df = df[df["Factory"].isin(sel_factories)]
if sel_divisions: df = df[df["Division"].isin(sel_divisions)]
if sel_modes:     df = df[df["Ship Mode"].isin(sel_modes)]

if df.empty:
    st.warning("??  No data matches the current filters. Please adjust the sidebar selections.")
    st.stop()

# -------------------------------------------------------------
# PAGE HEADER
# -------------------------------------------------------------
st.markdown("## ?? Nassau Candy Distributor")
st.markdown(
    "**Factory Reallocation & Shipping Optimization Recommendation System** "
    "� Predicts shipping outcomes � Recommends factory reassignments � Balances efficiency and profitability"
)
st.divider()

# -------------------------------------------------------------
# TABS
# -------------------------------------------------------------
TAB_LABELS = [
    "??  Overview & EDA",
    "??  Predictive Model",
    "??  Factory Simulator",
    "??  What-If Analysis",
    "??  Recommendations",
]
t1, t2, t3, t4, t5 = st.tabs(TAB_LABELS)


# -------------------------------------------------------------
#  TAB 1  �  OVERVIEW & EDA
# -------------------------------------------------------------
with t1:

    # -- KPI row --------------------------------------------
    st.markdown('<p class="sec-title">Key Performance Indicators</p>', unsafe_allow_html=True)
    k1, k2, k3, k4, k5, k6 = st.columns(6)
    kpis = [
        (k1, "Total Sales",      f"${df['Sales'].sum():,.0f}",            "10,194 orders"),
        (k2, "Gross Profit",     f"${df['Gross Profit'].sum():,.0f}",     f"{df['Profit Margin (%)'].mean():.1f}% margin"),
        (k3, "Avg Lead Time",    f"{df['Lead Time'].mean():.1f} days",    "across all ship modes"),
        (k4, "Total Units",      f"{int(df['Units'].sum()):,}",           "units shipped"),
        (k5, "Active Products",  str(df['Product Name'].nunique()),       "across 3 divisions"),
        (k6, "Regions Served",   str(df['Region'].nunique()),             "+ " + str(df['Country/Region'].nunique()) + " countries"),
    ]
    for col, label, val, sub in kpis:
        col.markdown(
            f'<div class="kpi-wrap"><p class="kpi-label">{label}</p>'
            f'<p class="kpi-value">{val}</p>'
            f'<p class="kpi-sub">{sub}</p></div>',
            unsafe_allow_html=True,
        )

    # -- Sales & Profit by Factory ---------------------------
    st.markdown('<p class="sec-title">Sales & Profitability by Factory</p>', unsafe_allow_html=True)
    c1, c2 = st.columns(2)

    with c1:
        fac_grp = (
            df.groupby("Factory")[["Sales", "Gross Profit", "Cost"]]
            .sum().reset_index()
            .melt(id_vars="Factory", var_name="Metric", value_name="Amount")
        )
        fig = px.bar(
            fac_grp, x="Factory", y="Amount", color="Metric",
            barmode="group",
            title="Sales vs Gross Profit vs Cost by Factory",
            color_discrete_map={"Sales": "#4f46e5", "Gross Profit": "#10b981", "Cost": "#ef4444"},
            labels={"Amount": "Amount (USD)", "Factory": ""},
        )
        fig.update_xaxes(tickangle=-15)
        st.plotly_chart(pl(fig), width='stretch')

    with c2:
        div_gp = df.groupby("Division")["Gross Profit"].sum().reset_index()
        fig2 = px.pie(
            div_gp, names="Division", values="Gross Profit",
            title="Gross Profit Share by Division",
            color="Division", color_discrete_map=D_COLOR, hole=0.42,
        )
        fig2.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(pl(fig2), width='stretch')

    # -- Lead Time Analysis ----------------------------------
    st.markdown('<p class="sec-title">Shipping & Lead Time Analysis</p>', unsafe_allow_html=True)
    c3, c4 = st.columns(2)

    with c3:
        lt_grp = df.groupby(["Ship Mode", "Region"])["Lead Time"].mean().reset_index()
        lt_grp["Lead Time"] = lt_grp["Lead Time"].round(2)
        fig3 = px.bar(
            lt_grp, x="Ship Mode", y="Lead Time", color="Region",
            barmode="group",
            title="Average Lead Time by Ship Mode & Region",
            color_discrete_map=R_COLOR,
            labels={"Lead Time": "Avg Lead Time (days)", "Ship Mode": ""},
        )
        fig3.update_xaxes(tickangle=-10)
        st.plotly_chart(pl(fig3), width='stretch')

    with c4:
        fig4 = px.box(
            df, x="Factory", y="Lead Time", color="Factory",
            title="Lead Time Distribution by Factory",
            color_discrete_map=F_COLOR,
            labels={"Lead Time": "Lead Time (days)", "Factory": ""},
        )
        fig4.update_layout(showlegend=False)
        fig4.update_xaxes(tickangle=-15)
        st.plotly_chart(pl(fig4), width='stretch')

    # -- Temporal Trends -------------------------------------
    st.markdown('<p class="sec-title">Temporal Sales Trends</p>', unsafe_allow_html=True)
    c5, c6 = st.columns(2)

    with c5:
        monthly = (
            df.groupby("Month Label")["Sales"]
            .sum().reset_index()
            .sort_values("Month Label")
        )
        fig5 = px.area(
            monthly, x="Month Label", y="Sales",
            title="Monthly Sales Trend (2024 � 2025)",
            labels={"Sales": "Total Sales (USD)", "Month Label": ""},
            color_discrete_sequence=["#4f46e5"],
        )
        fig5.update_xaxes(tickangle=-45, nticks=16)
        fig5.update_traces(line_width=2)
        st.plotly_chart(pl(fig5), width='stretch')

    with c6:
        mode_cnt = df["Ship Mode"].value_counts().reset_index()
        mode_cnt.columns = ["Ship Mode", "Count"]
        fig6 = px.bar(
            mode_cnt, x="Ship Mode", y="Count",
            title="Order Volume by Ship Mode",
            color="Ship Mode",
            color_discrete_map=M_COLOR,
            labels={"Count": "Number of Orders", "Ship Mode": ""},
        )
        fig6.update_layout(showlegend=False)
        st.plotly_chart(pl(fig6), width='stretch')

    # -- Product-level breakdown -----------------------------
    st.markdown('<p class="sec-title">Product Performance</p>', unsafe_allow_html=True)
    prod_grp = (
        df.groupby(["Product Name", "Division"])
        .agg(Sales=("Sales","sum"), GP=("Gross Profit","sum"), Orders=("Row ID","count"))
        .reset_index()
        .sort_values("GP", ascending=False)
    )
    prod_grp["Margin (%)"] = (prod_grp["GP"] / prod_grp["Sales"] * 100).round(1)
    fig7 = px.bar(
        prod_grp, x="GP", y="Product Name", color="Division",
        orientation="h",
        title="Gross Profit by Product",
        color_discrete_map=D_COLOR,
        labels={"GP": "Gross Profit (USD)", "Product Name": ""},
    )
    fig7.update_layout(height=420)
    st.plotly_chart(pl(fig7, h=420), width='stretch')

    # -- Factory map -----------------------------------------
    st.markdown('<p class="sec-title">Factory Locations</p>', unsafe_allow_html=True)
    map_rows = []
    for fac, (lat, lon) in FACTORY_COORDS.items():
        fdf = df_raw[df_raw["Factory"] == fac]
        map_rows.append({
            "Factory": fac,
            "lat": lat, "lon": lon,
            "Total Sales ($)":   round(fdf["Sales"].sum(), 0),
            "Avg Lead Time":     round(fdf["Lead Time"].mean(), 1),
            "Total Orders":      len(fdf),
        })
    map_df = pd.DataFrame(map_rows)
    fig_map = px.scatter_mapbox(
        map_df, lat="lat", lon="lon",
        hover_name="Factory",
        hover_data={"Total Sales ($)": ":,.0f", "Avg Lead Time": True,
                    "Total Orders": True, "lat": False, "lon": False},
        size="Total Sales ($)", color="Factory",
        color_discrete_map=F_COLOR,
        size_max=50, zoom=3.2,
        mapbox_style="carto-positron",
        title="Factory Locations  (bubble size = total sales)",
    )
    fig_map.update_layout(
        height=460,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=40, b=0),
    )
    st.plotly_chart(fig_map, width='stretch')

    # -- Raw data --------------------------------------------
    with st.expander("📊  View Raw / Filtered Data"):
        st.dataframe(df.reset_index(drop=True), width='stretch', height=320)
        st.download_button(
            "📥  Download Filtered CSV",
            df.to_csv(index=False).encode(),
            "nassau_filtered.csv", "text/csv",
        )


# -------------------------------------------------------------
#  TAB 2  �  PREDICTIVE MODEL
# -------------------------------------------------------------
with t2:
    st.markdown("### Predictive Modeling — Shipping Lead Time")
    st.info(
        "Three regression models are trained on **Ship Mode, Region, Factory, Division** "
        "to predict shipping lead time. The best-performing model (lowest RMSE) is used "
        "in the Simulator and Recommendations tabs.",
        icon="ℹ️",
    )

    # -- Model comparison cards ------------------------------
    st.markdown('<p class="sec-title">Model Evaluation Results</p>', unsafe_allow_html=True)
    mc1, mc2, mc3 = st.columns(3)
    for col, (name, res) in zip([mc1, mc2, mc3], model_results.items()):
        is_best = name == best_name
        extra   = " ? Best Model" if is_best else ""
        cls     = "mcard mcard-best" if is_best else "mcard"
        col.markdown(
            f'<div class="{cls}">'
            f'<div class="mcard-name">{name}{extra}</div>'
            f'<div class="mcard-stat">'
            f'RMSE : <b>{res["rmse"]}</b><br>'
            f'MAE  : <b>{res["mae"]}</b><br>'
            f'R²   : <b>{res["r2"]}</b>'
            f'</div></div>',
            unsafe_allow_html=True,
        )

    # -- Charts ---------------------------------------------
    st.markdown('<p class="sec-title">Performance Comparison</p>', unsafe_allow_html=True)
    pc1, pc2 = st.columns(2)

    with pc1:
        names = list(model_results.keys())
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="RMSE", x=names,
            y=[model_results[n]["rmse"] for n in names],
            marker_color="#4f46e5",
        ))
        fig_bar.add_trace(go.Bar(
            name="MAE", x=names,
            y=[model_results[n]["mae"] for n in names],
            marker_color="#10b981",
        ))
        fig_bar.update_layout(
            title="RMSE & MAE � All Models",
            barmode="group",
            **PLOTLY_LAYOUT, height=340,
        )
        st.plotly_chart(fig_bar, width='stretch')

    with pc2:
        r2_vals = [model_results[n]["r2"] for n in names]
        fig_r2  = px.bar(
            x=names, y=r2_vals,
            title="R� Score � All Models",
            color=names,
            color_discrete_sequence=["#4f46e5", "#10b981", "#f59e0b"],
            labels={"x": "Model", "y": "R� Score"},
        )
        fig_r2.update_layout(showlegend=False, **PLOTLY_LAYOUT, height=340)
        fig_r2.add_hline(y=1.0, line_dash="dot", line_color="gray",
                         annotation_text="Perfect = 1.0", annotation_position="top right")
        st.plotly_chart(fig_r2, width='stretch')

    # -- Actual vs Predicted ---------------------------------
    st.markdown(f'<p class="sec-title">Actual vs Predicted Lead Time � {best_name}</p>',
                unsafe_allow_html=True)
    y_te   = model_results[best_name]["y_test"]
    y_pr   = model_results[best_name]["preds"]
    sc_df  = pd.DataFrame({"Actual": y_te, "Predicted": y_pr.round(2)})
    fig_sc = px.scatter(
        sc_df, x="Actual", y="Predicted",
        opacity=0.55, color_discrete_sequence=["#4f46e5"],
        title=f"Actual vs Predicted � {best_name}",
        labels={"Actual": "Actual Lead Time (days)", "Predicted": "Predicted Lead Time (days)"},
    )
    mn, mx = 0, max(sc_df["Actual"].max(), sc_df["Predicted"].max()) + 1
    fig_sc.add_shape(type="line", x0=mn, y0=mn, x1=mx, y1=mx,
                     line=dict(color="#ef4444", dash="dash", width=1.5))
    fig_sc.add_annotation(x=mx - 0.5, y=mx - 0.5, text="Perfect fit",
                          showarrow=False, font=dict(color="#ef4444", size=11))
    st.plotly_chart(pl(fig_sc, h=400), width='stretch')

    # -- Feature Importance ----------------------------------
    if "Random Forest" in model_results:
        st.markdown('<p class="sec-title">Feature Importance (Random Forest)</p>', unsafe_allow_html=True)
        rf_mdl = model_results["Random Forest"]["model"]
        imp_df = pd.DataFrame(
            {"Feature": feat_cols, "Importance": rf_mdl.feature_importances_}
        ).sort_values("Importance")
        fig_imp = px.bar(
            imp_df, x="Importance", y="Feature", orientation="h",
            title="Feature Importance for Lead Time Prediction",
            color="Importance", color_continuous_scale="Blues",
            labels={"Importance": "Importance Score", "Feature": ""},
        )
        fig_imp.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT, height=280)
        st.plotly_chart(fig_imp, width='stretch')

    # -- Single prediction tool ------------------------------
    st.markdown('<p class="sec-title">Single Prediction Tool</p>', unsafe_allow_html=True)
    pp1, pp2, pp3, pp4 = st.columns(4)
    sp_mode    = pp1.selectbox("Ship Mode",  SHIP_MODES,  key="sp_mode")
    sp_region  = pp2.selectbox("Region",     REGIONS,     key="sp_region")
    sp_factory = pp3.selectbox("Factory",    FACTORIES,   key="sp_factory")
    sp_div     = pp4.selectbox("Division",   DIVISIONS,   key="sp_div")

    p_lead    = pred_lead(best_mdl, le_map, feat_cols, sp_mode, sp_region, sp_factory, sp_div)
    rule_lead = SHIP_BASE[sp_mode] + FACTORY_REGION_DIST[sp_factory][sp_region]

    pr1, pr2, pr3 = st.columns(3)
    pr1.metric("??  Predicted Lead Time",   f"{p_lead} days")
    pr2.metric("??  Rule-Based Lead Time",  f"{rule_lead} days")
    pr3.metric("?  Model vs Rule",
               f"{p_lead - rule_lead:+.2f} days",
               delta_color="inverse" if p_lead > rule_lead else "normal")


# -------------------------------------------------------------
#  TAB 3  �  FACTORY SIMULATOR
# -------------------------------------------------------------
with t3:
    st.markdown("### Factory Optimization Simulator")
    st.markdown(
        "Select a **product, destination region and ship mode** to compare predicted "
        "performance across all five factories and find the optimal assignment."
    )

    products_list = sorted(df_raw["Product Name"].unique())
    sc1, sc2, sc3 = st.columns(3)
    sim_prod    = sc1.selectbox("Product",   products_list, key="sim_prod")
    sim_region  = sc2.selectbox("Region",    REGIONS,       key="sim_region")
    sim_mode    = sc3.selectbox("Ship Mode", SHIP_MODES,    key="sim_mode")

    cur_factory = PRODUCT_FACTORY.get(sim_prod, FACTORIES[0])
    sim_div     = df_raw[df_raw["Product Name"] == sim_prod]["Division"].iloc[0]

    # Build per-factory predictions
    sim_rows = []
    for fac in FACTORIES:
        p_lt  = pred_lead(best_mdl, le_map, feat_cols, sim_mode, sim_region, fac, sim_div)
        r_lt  = SHIP_BASE[sim_mode] + FACTORY_REGION_DIST[fac][sim_region]
        fac_p = df_raw[(df_raw["Product Name"] == sim_prod) & (df_raw["Factory"] == fac)]
        if len(fac_p) == 0:
            fac_p = df_raw[df_raw["Product Name"] == sim_prod]
        avg_gp = fac_p["Gross Profit"].mean()
        avg_sl = fac_p["Sales"].mean()
        sim_rows.append({
            "Factory":               fac,
            "Is Current":            fac == cur_factory,
            "Predicted Lead (days)": p_lt,
            "Rule-Based Lead (days)":r_lt,
            "Avg Gross Profit ($)":  round(avg_gp, 2),
            "Avg Sales ($)":         round(avg_sl, 2),
            "Historical Orders":     len(df_raw[(df_raw["Product Name"]==sim_prod)&(df_raw["Factory"]==fac)]),
        })
    sim_df  = pd.DataFrame(sim_rows).sort_values("Predicted Lead (days)").reset_index(drop=True)
    best_fac = sim_df.iloc[0]["Factory"]
    cur_row  = sim_df[sim_df["Is Current"]].iloc[0]
    lead_save = round(float(cur_row["Predicted Lead (days)"]) - float(sim_df.iloc[0]["Predicted Lead (days)"]), 2)
    gp_delta  = round(float(sim_df.iloc[0]["Avg Gross Profit ($)"]) - float(cur_row["Avg Gross Profit ($)"]), 2)

    # KPIs
    sm1, sm2, sm3, sm4 = st.columns(4)
    sm1.metric("Current Factory",       cur_factory)
    sm2.metric("Best Alternative",      best_fac,
               delta="No change needed" if best_fac == cur_factory else "? Switch recommended")
    sm3.metric("Lead Time Saving",      f"{lead_save} days",
               delta=f"{lead_save} days faster" if lead_save > 0 else "Already optimal",
               delta_color="normal" if lead_save > 0 else "off")
    sm4.metric("GP Impact ($)",         f"${gp_delta:+.2f}",
               delta_color="normal" if gp_delta >= 0 else "inverse")

    # Charts
    st.markdown('<p class="sec-title">Factory Performance Comparison</p>', unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)

    with sc1:
        col_list = [F_COLOR.get(f, "#888") for f in sim_df["Factory"]]
        fig_sl = go.Figure()
        fig_sl.add_trace(go.Bar(
            x=sim_df["Factory"],
            y=sim_df["Predicted Lead (days)"],
            marker_color=col_list,
            text=sim_df["Predicted Lead (days)"].apply(lambda v: f"{v}d"),
            textposition="outside",
        ))
        fig_sl.add_hline(
            y=float(cur_row["Predicted Lead (days)"]),
            line_dash="dash", line_color="#ef4444",
            annotation_text=f"Current: {cur_factory}",
            annotation_position="top right",
        )
        fig_sl.update_layout(
            title="Predicted Lead Time per Factory",
            yaxis_title="Lead Time (days)",
            **PLOTLY_LAYOUT, height=360,
        )
        fig_sl.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_sl, width='stretch')

    with sc2:
        fig_gp = go.Figure()
        fig_gp.add_trace(go.Bar(
            x=sim_df["Factory"],
            y=sim_df["Avg Gross Profit ($)"],
            marker_color=col_list,
            text=sim_df["Avg Gross Profit ($)"].apply(lambda v: f"${v:.2f}"),
            textposition="outside",
        ))
        fig_gp.update_layout(
            title="Avg Gross Profit per Order",
            yaxis_title="Gross Profit (USD)",
            **PLOTLY_LAYOUT, height=360,
        )
        fig_gp.update_xaxes(tickangle=-15)
        st.plotly_chart(fig_gp, width='stretch')

    # Scorecard table
    st.markdown('<p class="sec-title">Detailed Factory Scorecard</p>', unsafe_allow_html=True)
    sim_df.insert(0, "Rank", range(1, len(sim_df) + 1))
    sim_df["Status"] = sim_df.apply(
        lambda r: "? Recommended" if r["Factory"] == best_fac and not r["Is Current"]
                  else ("?? Current" if r["Is Current"] else "�"),
        axis=1,
    )
    st.dataframe(
        sim_df[["Rank","Factory","Status","Predicted Lead (days)",
                "Rule-Based Lead (days)","Avg Gross Profit ($)","Avg Sales ($)","Historical Orders"]],
        hide_index=True, width='stretch',
    )

    # Radar chart
    st.markdown('<p class="sec-title">Multi-Metric Radar Comparison</p>', unsafe_allow_html=True)
    max_lt  = float(sim_df["Predicted Lead (days)"].max())
    max_gp  = float(sim_df["Avg Gross Profit ($)"].max()) or 1.0
    max_ord = float(sim_df["Historical Orders"].max()) or 1.0
    cats    = ["Speed Score", "Profit Score", "Volume Score", "Efficiency"]
    fig_rad = go.Figure()
    for _, row in sim_df.iterrows():
        sp  = round((max_lt - row["Predicted Lead (days)"]) / max_lt * 10, 2)
        pr  = round(row["Avg Gross Profit ($)"] / max_gp * 10, 2)
        vo  = round(row["Historical Orders"] / max_ord * 10, 2)
        ef  = round((sp + pr) / 2, 2)
        fig_rad.add_trace(go.Scatterpolar(
            r=[sp, pr, vo, ef, sp],
            theta=cats + [cats[0]],
            name=row["Factory"],
            line_color=F_COLOR.get(row["Factory"], "#888"),
            fill="toself", opacity=0.25,
        ))
    fig_rad.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 10])),
        height=430,
        paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", y=-0.12),
    )
    st.plotly_chart(fig_rad, width='stretch')


# -------------------------------------------------------------
#  TAB 4  �  WHAT-IF ANALYSIS
# -------------------------------------------------------------
with t4:
    st.markdown("### What-If Scenario Analysis")
    st.markdown(
        "Simulate different factory configurations. Adjust the **Speed ? Profit priority** "
        "in the sidebar and the scenario filters below to explore trade-offs."
    )

    wc1, wc2 = st.columns(2)
    wi_region = wc1.selectbox("Target Region",    ["All"] + REGIONS,    key="wi_region")
    wi_mode   = wc2.selectbox("Ship Mode Filter", ["All"] + SHIP_MODES, key="wi_mode")

    tgt_region = wi_region if wi_region != "All" else "Interior"
    tgt_mode   = wi_mode   if wi_mode   != "All" else "Standard Class"

    # Current avg lead per factory (from filtered data)
    cur_leads = {}
    for fac in FACTORIES:
        sub = df[df["Factory"] == fac]["Lead Time"]
        cur_leads[fac] = float(sub.mean()) if len(sub) else 0.0

    # Compute recommended lead per factory (best product assignment)
    speed_w  = priority / 100.0
    profit_w = 1.0 - speed_w

    # Per-product recommendation engine
    prod_recs = []
    for prod in df_raw["Product Name"].unique():
        prows   = df_raw[df_raw["Product Name"] == prod]
        cur_fac = PRODUCT_FACTORY.get(prod, FACTORIES[0])
        pdiv    = prows["Division"].iloc[0]

        best_score = -np.inf
        best_fac_r = cur_fac
        fac_details = {}
        for fac in FACTORIES:
            lt  = pred_lead(best_mdl, le_map, feat_cols, tgt_mode, tgt_region, fac, pdiv)
            gp  = df_raw[df_raw["Factory"] == fac]["Gross Profit"].mean()
            gp  = gp if not np.isnan(gp) else 0.0
            sp  = (10 - lt) / 10.0
            prs = gp / (df_raw["Gross Profit"].max() or 1.0)
            sc  = speed_w * sp + profit_w * prs
            fac_details[fac] = {"lt": lt, "gp": gp, "score": sc}
            if fac != cur_fac and sc > best_score:
                best_score = sc
                best_fac_r = fac

        cur_lt  = fac_details[cur_fac]["lt"]
        best_lt = fac_details[best_fac_r]["lt"]
        cur_gp  = df_raw[df_raw["Factory"] == cur_fac]["Gross Profit"].mean()
        best_gp = df_raw[df_raw["Factory"] == best_fac_r]["Gross Profit"].mean()
        cur_gp  = cur_gp  if not np.isnan(cur_gp)  else 0.0
        best_gp = best_gp if not np.isnan(best_gp) else 0.0

        delta_lt = round(cur_lt - best_lt, 2)    # positive = saving
        gp_chg   = round(best_gp - cur_gp, 2)
        conf     = round(min(99, 50 + abs(delta_lt) * 9 + speed_w * 14))
        if abs(gp_chg) < 1 and delta_lt >= 0:
            risk = "low"
        elif abs(gp_chg) < 3:
            risk = "medium"
        else:
            risk = "high"

        prod_recs.append({
            "Product":              prod,
            "Division":             pdiv,
            "Current Factory":      cur_fac,
            "Recommended Factory":  best_fac_r,
            "Current Lead (days)":  cur_lt,
            "Rec Lead (days)":      best_lt,
            "Lead Saving (days)":   delta_lt,
            "GP Impact ($)":        gp_chg,
            "Confidence (%)":       conf,
            "Risk":                 risk,
            "Changed":              best_fac_r != cur_fac,
        })

    rec_df = pd.DataFrame(prod_recs)

    # -- Current vs Recommended lead chart ------------------
    st.markdown('<p class="sec-title">Current vs Recommended Average Lead Time by Factory</p>',
                unsafe_allow_html=True)
    rec_leads_by_fac = {
        fac: pred_lead(best_mdl, le_map, feat_cols, tgt_mode, tgt_region, fac, "Chocolate")
        for fac in FACTORIES
    }
    fig_wi = go.Figure()
    fig_wi.add_trace(go.Bar(
        x=FACTORIES,
        y=[cur_leads[f] for f in FACTORIES],
        name="Current Avg Lead",
        marker_color="#94a3b8",
        text=[f"{cur_leads[f]:.1f}d" for f in FACTORIES],
        textposition="outside",
    ))
    fig_wi.add_trace(go.Bar(
        x=FACTORIES,
        y=[rec_leads_by_fac[f] for f in FACTORIES],
        name="Predicted Optimal Lead",
        marker_color="#4f46e5",
        text=[f"{rec_leads_by_fac[f]:.1f}d" for f in FACTORIES],
        textposition="outside",
    ))
    fig_wi.update_layout(
        barmode="group",
        yaxis_title="Lead Time (days)",
        **PLOTLY_LAYOUT, height=380,
    )
    fig_wi.update_xaxes(tickangle=-15)
    st.plotly_chart(fig_wi, width='stretch')

    # -- Efficiency gain KPIs --------------------------------
    st.markdown('<p class="sec-title">Projected Efficiency Gains</p>', unsafe_allow_html=True)
    changed = rec_df[rec_df["Changed"]]
    eg1, eg2, eg3, eg4 = st.columns(4)
    eg1.metric("Products to Reassign",    len(changed))
    eg2.metric("Avg Lead Time Saving",    f"{changed['Lead Saving (days)'].mean():.1f} days"
               if len(changed) else "0 days")
    eg3.metric("Total GP Impact",         f"${changed['GP Impact ($)'].sum():+.2f}"
               if len(changed) else "$0.00")
    eg4.metric("Avg Model Confidence",    f"{rec_df['Confidence (%)'].mean():.0f}%")

    # -- Per-product lead saving & GP impact ----------------
    st.markdown('<p class="sec-title">Per-Product Impact</p>', unsafe_allow_html=True)
    wc3, wc4 = st.columns(2)
    ch_sorted = changed.sort_values("Lead Saving (days)", ascending=True)

    with wc3:
        if len(ch_sorted):
            fig_ld = px.bar(
                ch_sorted, x="Lead Saving (days)", y="Product",
                orientation="h",
                title="Lead Time Reduction per Product",
                color="Lead Saving (days)",
                color_continuous_scale="Blues",
                labels={"Lead Saving (days)": "Days Saved", "Product": ""},
            )
            fig_ld.update_layout(coloraxis_showscale=False, **PLOTLY_LAYOUT,
                                 height=max(300, len(ch_sorted)*38 + 60))
            st.plotly_chart(fig_ld, width='stretch')

    with wc4:
        if len(ch_sorted):
            gp_colors = ["#10b981" if v >= 0 else "#ef4444"
                         for v in ch_sorted["GP Impact ($)"]]
            fig_gpi = go.Figure(go.Bar(
                x=ch_sorted["Product"],
                y=ch_sorted["GP Impact ($)"],
                marker_color=gp_colors,
                text=ch_sorted["GP Impact ($)"].apply(lambda v: f"${v:+.2f}"),
                textposition="outside",
            ))
            fig_gpi.update_layout(
                title="Gross Profit Impact per Product",
                yaxis_title="GP Change (USD)",
                **PLOTLY_LAYOUT,
                height=max(300, len(ch_sorted)*38 + 60),
            )
            fig_gpi.update_xaxes(tickangle=-40)
            st.plotly_chart(fig_gpi, width='stretch')

    # -- Priority trade-off curve ----------------------------
    st.markdown('<p class="sec-title">Speed vs Profit Trade-Off Curve</p>', unsafe_allow_html=True)
    prange  = list(range(0, 101, 10))
    tc_rows = []
    for pv in prange:
        sw, pw = pv / 100, 1 - pv / 100
        tot_save = 0.0
        tot_gpi  = 0.0
        for prod in df_raw["Product Name"].unique():
            cf   = PRODUCT_FACTORY.get(prod, FACTORIES[0])
            pdv  = df_raw[df_raw["Product Name"] == prod]["Division"].iloc[0]
            c_lt = pred_lead(best_mdl, le_map, feat_cols, tgt_mode, tgt_region, cf, pdv)
            c_gp = df_raw[df_raw["Factory"] == cf]["Gross Profit"].mean()
            c_gp = c_gp if not np.isnan(c_gp) else 0.0
            bsc  = -np.inf
            b_lt = c_lt
            b_gp = c_gp
            for fac in FACTORIES:
                if fac == cf:
                    continue
                lt  = pred_lead(best_mdl, le_map, feat_cols, tgt_mode, tgt_region, fac, pdv)
                gp  = df_raw[df_raw["Factory"] == fac]["Gross Profit"].mean()
                gp  = gp if not np.isnan(gp) else 0.0
                sc  = sw * (10 - lt) / 10 + pw * (gp / (df_raw["Gross Profit"].max() or 1))
                if sc > bsc:
                    bsc = sc
                    b_lt = lt
                    b_gp = gp
            tot_save += (c_lt - b_lt)
            tot_gpi  += (b_gp - c_gp)
        tc_rows.append({
            "Speed Priority (%)": pv,
            "Total Lead Saving (days)": round(tot_save, 1),
            "Total GP Impact ($)":      round(tot_gpi, 2),
        })
    tc_df = pd.DataFrame(tc_rows)

    fig_tc = make_subplots(specs=[[{"secondary_y": True}]])
    fig_tc.add_trace(
        go.Scatter(x=tc_df["Speed Priority (%)"], y=tc_df["Total Lead Saving (days)"],
                   name="Lead Saving (days)", mode="lines+markers",
                   line=dict(color="#4f46e5", width=2)), secondary_y=False,
    )
    fig_tc.add_trace(
        go.Scatter(x=tc_df["Speed Priority (%)"], y=tc_df["Total GP Impact ($)"],
                   name="GP Impact ($)", mode="lines+markers",
                   line=dict(color="#10b981", width=2, dash="dash")), secondary_y=True,
    )
    fig_tc.add_vline(x=priority, line_dash="dot", line_color="#ef4444",
                     annotation_text=f"Current priority: {priority}%",
                     annotation_position="top left")
    fig_tc.update_xaxes(title_text="Speed Priority (%)")
    fig_tc.update_yaxes(title_text="Total Lead Saving (days)", secondary_y=False,
                        gridcolor="rgba(128,128,128,.12)")
    fig_tc.update_yaxes(title_text="Total GP Impact ($)", secondary_y=True)
    fig_tc.update_layout(
        title="Speed vs Profit Trade-Off Across All Priority Settings",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=400,
        legend=dict(orientation="h", y=-0.18),
        font=dict(family="Inter, Segoe UI, sans-serif", size=12),
    )
    st.plotly_chart(fig_tc, width='stretch')


# -------------------------------------------------------------
#  TAB 5  �  RECOMMENDATIONS
# -------------------------------------------------------------
with t5:
    st.markdown("### Factory Reassignment Recommendations")
    st.markdown(
        f"Ranked by composite **lead-time reduction + profit impact** score "
        f"at the current priority setting (**{priority}% speed / {100-priority}% profit**). "
        f"Showing top **{top_n}** recommendations."
    )

    # Score + sort
    max_save = rec_df["Lead Saving (days)"].abs().max() or 1
    max_gpi  = rec_df["GP Impact ($)"].abs().max() or 1
    rec_df["Score"] = (
        (priority / 100) * rec_df["Lead Saving (days)"].clip(lower=0) / max_save * 10
        + (1 - priority / 100) * rec_df["GP Impact ($)"] / max_gpi * 10
    )
    top_df = (
        rec_df[rec_df["Changed"]]
        .sort_values("Score", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )

    # -- Summary KPIs -------------------------------------
    st.markdown('<p class="sec-title">Recommendation Summary</p>', unsafe_allow_html=True)
    rk1, rk2, rk3, rk4 = st.columns(4)
    rk1.metric("Recommendations",      len(top_df))
    rk2.metric("Low Risk",   int((top_df["Risk"]=="low").sum()),
               delta="? Safe to action", delta_color="off")
    rk3.metric("Medium Risk",int((top_df["Risk"]=="medium").sum()),
               delta_color="off")
    rk4.metric("High Risk",  int((top_df["Risk"]=="high").sum()),
               delta="?? Review carefully", delta_color="inverse")

    # -- Recommendations Table -----------------------------
    st.markdown('<p class="sec-title">Ranked Reassignment Table</p>', unsafe_allow_html=True)
    tbl_rows = ""
    for idx, row in top_df.iterrows():
        gp_col = "#10b981" if row["GP Impact ($)"] >= 0 else "#ef4444"
        ld_col = "#4f46e5" if row["Lead Saving (days)"] > 0 else "#94a3b8"
        tbl_rows += f"""
        <tr>
          <td><b>{idx+1}</b></td>
          <td>{row['Product']}</td>
          <td>{row['Division']}</td>
          <td><span class="badge b-cur">{row['Current Factory']}</span></td>
          <td><span class="badge b-rec">{row['Recommended Factory']}</span></td>
          <td style="color:{ld_col};font-weight:600">
            {row['Current Lead (days)']:.1f}d ? {row['Rec Lead (days)']:.1f}d
          </td>
          <td style="color:{ld_col};font-weight:700">
            {'-' if row['Lead Saving (days)']>0 else '+'}{abs(row['Lead Saving (days)']):.1f}d
          </td>
          <td style="color:{gp_col};font-weight:600">${row['GP Impact ($)']:+.2f}</td>
          <td>{row['Confidence (%)']:.0f}%</td>
          <td>{risk_badge(row['Risk'])}</td>
        </tr>"""

    st.markdown(f"""
    <div style="overflow-x:auto;border:1px solid rgba(128,128,128,.15);border-radius:10px;padding:8px">
    <table class="rtable">
      <thead>
        <tr>
          <th>#</th><th>Product</th><th>Division</th>
          <th>Current Factory</th><th>Recommended Factory</th>
          <th>Lead Time</th><th>Lead Saving</th>
          <th>GP Impact</th><th>Confidence</th><th>Risk</th>
        </tr>
      </thead>
      <tbody>{tbl_rows}</tbody>
    </table>
    </div>
    """, unsafe_allow_html=True)

    # -- Charts -------------------------------------------
    st.markdown('<p class="sec-title">Recommendation Analytics</p>', unsafe_allow_html=True)
    ra1, ra2 = st.columns(2)

    with ra1:
        fig_conf = px.bar(
            top_df.sort_values("Confidence (%)"),
            x="Confidence (%)", y="Product",
            orientation="h",
            title="Confidence Score by Product",
            color="Confidence (%)",
            color_continuous_scale="Purples",
            labels={"Product": ""},
        )
        fig_conf.update_layout(
            coloraxis_showscale=False,
            **PLOTLY_LAYOUT,
            height=max(320, len(top_df) * 40 + 60),
        )
        st.plotly_chart(fig_conf, width='stretch')

    with ra2:
        risk_cnt = top_df["Risk"].value_counts().reset_index()
        risk_cnt.columns = ["Risk", "Count"]
        fig_risk = px.pie(
            risk_cnt, names="Risk", values="Count",
            title="Risk Distribution",
            color="Risk",
            color_discrete_map={"low":"#10b981","medium":"#f59e0b","high":"#ef4444"},
            hole=0.42,
        )
        fig_risk.update_traces(textposition="outside", textinfo="percent+label")
        st.plotly_chart(pl(fig_risk, h=340), width='stretch')

    # -- Bubble chart -------------------------------------
    st.markdown('<p class="sec-title">Lead Saving vs GP Impact � Confidence Map</p>',
                unsafe_allow_html=True)
    fig_bub = px.scatter(
        top_df,
        x="Lead Saving (days)", y="GP Impact ($)",
        size="Confidence (%)", color="Risk",
        hover_name="Product",
        hover_data={"Current Factory": True, "Recommended Factory": True,
                    "Confidence (%)": True, "Division": True},
        color_discrete_map={"low":"#10b981","medium":"#f59e0b","high":"#ef4444"},
        title="Lead Time Saving vs GP Impact  (bubble size = confidence)",
        size_max=50,
        labels={"Lead Saving (days)":"Lead Time Saved (days)","GP Impact ($)":"GP Change (USD)"},
    )
    fig_bub.add_hline(y=0, line_dash="dash", line_color="rgba(128,128,128,.4)")
    fig_bub.add_vline(x=0, line_dash="dash", line_color="rgba(128,128,128,.4)")
    st.plotly_chart(pl(fig_bub, h=430), width='stretch')

    # -- Sankey flow --------------------------------------
    st.markdown('<p class="sec-title">Product Reallocation Flow</p>', unsafe_allow_html=True)
    sdf = (
        top_df.groupby(["Current Factory", "Recommended Factory"])
        .size().reset_index(name="Count")
    )
    all_nodes  = list(dict.fromkeys(
        sdf["Current Factory"].tolist() + sdf["Recommended Factory"].tolist()
    ))
    nidx       = {n: i for i, n in enumerate(all_nodes)}
    node_colors = [F_COLOR.get(n, "#888") for n in all_nodes]
    fig_sk = go.Figure(go.Sankey(
        node=dict(label=all_nodes, color=node_colors, pad=20, thickness=22),
        link=dict(
            source=[nidx[r["Current Factory"]]     for _, r in sdf.iterrows()],
            target=[nidx[r["Recommended Factory"]] for _, r in sdf.iterrows()],
            value= sdf["Count"].tolist(),
            color=["rgba(79,70,229,.28)"] * len(sdf),
        ),
    ))
    fig_sk.update_layout(
        title="Factory Reallocation Flow (Sankey Diagram)",
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", size=13),
    )
    st.plotly_chart(fig_sk, width='stretch')

    # -- Risk panel ---------------------------------------
    st.markdown('<p class="sec-title">Risk & Impact Panel</p>', unsafe_allow_html=True)
    rp1, rp2, rp3 = st.columns(3)
    for col, level, bg, border, icon in [
        (rp1, "low",    "#d1fae5", "#059669", "?"),
        (rp2, "medium", "#fef3c7", "#d97706", "??"),
        (rp3, "high",   "#fee2e2", "#dc2626", "??"),
    ]:
        items = top_df[top_df["Risk"] == level]
        li_html = "".join(
            f"<li>{r['Product']} "
            f"<span style='color:#4f46e5'>? {r['Recommended Factory']}</span></li>"
            for _, r in items.iterrows()
        ) if len(items) else "<li style='opacity:.5'>None</li>"
        col.markdown(
            f'<div class="rpanel" style="background:{bg};border:1.5px solid {border}">'
            f'<div class="rpanel-title">{icon} {level.upper()} Risk � {len(items)}</div>'
            f'<ul>{li_html}</ul></div>',
            unsafe_allow_html=True,
        )

    # -- Exports ------------------------------------------
    st.markdown('<p class="sec-title">Export Results</p>', unsafe_allow_html=True)
    ex1, ex2, ex3 = st.columns(3)
    ex1.download_button(
        "??  Top Recommendations CSV",
        top_df.drop(columns=["Changed","Score"], errors="ignore").to_csv(index=False).encode(),
        "nassau_top_recommendations.csv", "text/csv",
    )
    ex2.download_button(
        "??  Full Product Analysis CSV",
        rec_df.drop(columns=["Changed","Score"], errors="ignore").to_csv(index=False).encode(),
        "nassau_full_analysis.csv", "text/csv",
    )
    ex3.download_button(
        "??  Filtered Dataset CSV",
        df.to_csv(index=False).encode(),
        "nassau_filtered_dataset.csv", "text/csv",
    )

    # -- Executive Summary ---------------------------------
    st.markdown('<p class="sec-title">Executive Summary</p>', unsafe_allow_html=True)
    low_c  = int((top_df["Risk"]=="low").sum())
    med_c  = int((top_df["Risk"]=="medium").sum())
    high_c = int((top_df["Risk"]=="high").sum())
    best_p = top_df.loc[top_df["Lead Saving (days)"].idxmax(), "Product"] if len(top_df) else "N/A"
    avg_c  = top_df["Confidence (%)"].mean() if len(top_df) else 0
    best_m = model_results[best_name]

    st.markdown(f"""
    <div class="exec-box">
    <p><b>Dataset:</b> 10,194 orders � 15 products � 5 factories � 4 regions � 2 countries (2024�2025)</p>

    <p><b>Best Predictive Model:</b> {best_name} &nbsp;|&nbsp;
    RMSE = {best_m['rmse']} &nbsp;|&nbsp; MAE = {best_m['mae']} &nbsp;|&nbsp; R� = {best_m['r2']}</p>

    <p><b>Optimization Priority:</b> {priority}% speed / {100-priority}% profit �
    {len(top_df)} reassignment opportunities identified:
    <span style="color:#059669"><b>{low_c} low risk</b></span>,
    <span style="color:#d97706"><b>{med_c} medium risk</b></span>,
    <span style="color:#dc2626"><b>{high_c} high risk</b></span>.</p>

    <p><b>Greatest lead time improvement:</b> {best_p}.
    Average confidence score across all recommendations: <b>{avg_c:.0f}%</b>.</p>

    <p>This system elevates Nassau Candy Distributor from descriptive analytics to
    <b>intelligent decision-making</b> � combining predictive modelling with optimisation logic
    to improve shipping efficiency without sacrificing profitability.</p>
    </div>
    """, unsafe_allow_html=True)


