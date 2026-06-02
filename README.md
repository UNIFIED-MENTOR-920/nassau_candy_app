# Nassau Candy Distributor
## Factory Reallocation & Shipping Optimization System

### How to Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Deploy on Streamlit Community Cloud
1. Push this folder to a GitHub repository
2. Go to https://share.streamlit.io
3. Click "New app" → select your repo → set `app.py` as main file
4. Click Deploy

### Files
- `app.py` — Main Streamlit application
- `Nassau_Candy_Distributor.csv` — Dataset (10,194 orders, 18 fields)
- `requirements.txt` — Python dependencies

### Dashboard Tabs
1. **Overview & EDA** — KPIs, sales charts, lead time analysis, factory map
2. **Predictive Model** — Linear Regression, Random Forest, Gradient Boosting comparison
3. **Factory Simulator** — Per-product factory performance comparison
4. **What-If Analysis** — Scenario simulation with speed vs profit slider
5. **Recommendations** — Ranked factory reassignment recommendations with risk & Sankey flow
