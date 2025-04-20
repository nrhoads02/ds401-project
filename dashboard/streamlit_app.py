# app.py
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime

from src.data_transformation.transformation_pipeline import transformation_pipeline
from src.data_extraction.dataframe_loader import load_data
from src.data_modeling.lgbm_modeling import load_lgbm_model, predict_for_visualization

# -----------------------------------------------------------------------------
# About text with proper LaTeX for math
# -----------------------------------------------------------------------------
ABOUT_TEXT = r"""
### What Is an Option and Why Does Volatility Matter?  
An **option** is a contract giving the right—but not the obligation—to buy or sell a stock at a fixed price on or before a certain date.  
Options are priced largely based on **volatility**, which measures how much the stock’s price moves. Higher volatility raises the chance of large price swings, making options more expensive.

---

### Implied vs. Local Volatility Surfaces  
- **Implied volatility** is the market’s expectation of future volatility, backed out of option prices across strikes and expiries.  
- A **local volatility surface** goes deeper: it asks  
  > *“What instantaneous volatility at each stock price and future date would make today’s option prices consistent under a risk‑neutral model?”*  
  One inverts the option pricing model (Dupire’s formula) to recover this surface, which is essential for pricing more complex derivatives.

---

### Realized Local Volatility Surface  
The **Realized Local Volatility Surface (RLVS)** brings these ideas to historical data:

1. **Realized volatility** is what actually happened: how much the stock price varied over a past window.  
2. A simple realized‑vol curve is only a single line for the one path the stock took.  
3. The RLVS estimates the conditional expectation  
   $$
   \sigma_{\mathrm{real}}(K, T)
   \;=\;
   \mathbb{E}\bigl[\mathrm{RealizedVol}\,\bigm|\tfrac{S_T}{S_0}=e^{k}\bigr],
   $$
   where $K = S_0 e^{k}$ is the strike and $T$ is the time to expiry.  
   It tells you,  
   > *“Had the stock ended at strike \(K\) in \(T\) days, what volatility would we have realized?”*

---

This app lets you compare that **historical conditional vol surface** to today’s **market‑implied** local vol surface—highlighting where reality and expectation diverge.
"""

# -----------------------------------------------------------------------------
def generate_vol_surface(df: pl.DataFrame, stock: str, date, show_surface=False):
    """
    Generate a Realized Local Volatility surface.

    X-axis: Strike Price  
    Y-axis: Calendar Days to expiry  
    Z-axis: Annualized volatility
    """
    # 1) Select the single row for stock/date (include close)
    row = df.filter((pl.col("act_symbol") == stock) & (pl.col("date") == date))
    if row.height == 0:
        raise ValueError(f"No data found for {stock} on {date}")
    # get spot price S0
    S0 = float(row["close"][0])

    # 2) trading→calendar days
    trading_windows = [10, 15, 20, 25, 30, 35]
    cal_days = [int(round(w * 7/5)) for w in trading_windows]

    # 3) build realized_data dict
    realized_data = {}
    for w, cd in zip(trading_windows, cal_days):
        yrs = cd/365.0
        vol = float(row[f"YZVol_{w}_future"][0]) * np.sqrt(252)
        k   = float(row[f"LogPriceRatio_{w}_future"][0])
        skew = float(row[f"VolSkew_{w}_future"][0])
        curv = float(row[f"VolCurvature_{w}_future"][0])
        wing = float(row[f"WingRatio_{w}_future"][0])
        mr   = float(row[f"MeanReversion_{w}_future"][0])
        vov  = float(row[f"VolOfVol_{w}_future"][0])
        pvc  = float(row[f"PriceVolCorr_{w}_future"][0])
        vi   = float(row[f"VolIntensity_{w}_future"][0])
        realized_data[cd] = {
            "vol": vol,
            "k":   k,
            "skew": skew,
            "curv": curv,
            "wing": wing,
            "mr":   mr,
            "vov":  vov,
            "pvc":  pvc,
            "vi":   vi,
            "yrs":  yrs
        }

    # 4) dynamic moneyness grid
    ks_real = [d["k"] for d in realized_data.values()]
    k_min, k_max = min(ks_real)-0.15, max(ks_real)+0.15
    k_grid = np.linspace(k_min, k_max, 50)

    # 5) mesh for strikes
    Km, Tm = np.meshgrid(k_grid, cal_days)
    # convert log-moneyness → strike
    Strike_mesh = S0 * np.exp(Km)

    # 6) compute surface
    surf = np.zeros_like(Strike_mesh)
    for i, cd in enumerate(cal_days):
        data = realized_data[cd]
        vol0 = data["vol"]; yrs = data["yrs"]; sqrt_yrs = np.sqrt(yrs)
        for j, k in enumerate(k_grid):
            rel = k - data["k"]
            rel_s = rel/(vol0*sqrt_yrs)
            bf = min(yrs*5, 0.8)
            ek = (1-bf)*rel + bf*rel_s
            skew = data["skew"] * (abs(ek)*data["wing"] if ek<0 else -ek/data["wing"])
            curv = data["curv"] * ek**2 * (1+2*data["vov"])
            corr = data["pvc"] * ek * 0.5
            inten= (data["vi"]-0.5)*2*abs(ek)**2
            vol_lvl = 0.05*data["vov"]*yrs
            total = skew+curv+corr+inten
            v = vol0*(1+total)*(1+vol_lvl)
            surf[i,j] = max(0.05, min(v,2.0))

    # 7) actual realized markers
    strikes_obs = [S0*np.exp(d["k"]) for d in realized_data.values()]
    vols_obs    = [d["vol"] for d in realized_data.values()]

    # 8) Plotly
    fig = go.Figure()
    fig.add_trace(go.Surface(
        x=Strike_mesh, y=Tm, z=surf,
        colorscale="Viridis", colorbar=dict(title="σ (ann.)"), name="Surface"
    ))
    fig.add_trace(go.Scatter3d(
        x=strikes_obs, y=cal_days, z=vols_obs,
        mode="markers", marker=dict(color="red", size=6),
        name="Realized"
    ))
    fig.update_layout(
        title=f"{stock} Realized Vol Surface – {date}",
        scene=dict(
            xaxis_title="Strike Price",
            yaxis_title="Calendar Days",
            zaxis_title="Annualized Volatility"
        ),
        width=900, height=700
    )
    if show_surface:
        fig.show()

    return fig, {
        "K": Strike_mesh, "T": Tm, "surface": surf,
        "cal_days": cal_days
    }

def display_surface_details(stock, surface, stype):
    with st.expander(f"{stype} Surface Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"**Stock:** {stock}")
            st.write(f"**Surface Type:** {stype}")
        with col2:
            st.write("**Trading → Calendar**")
            st.json({f"{t} td": f"{c} cd"
                     for t,c in zip(surface.get("trading_windows", []), surface["cal_days"])})

def main():
    st.set_page_config(page_title="Volatility Surface App", layout="wide")

    # Tabs for Visualizer and About
    viz_tab, about_tab = st.tabs(["Visualizer", "About"])

    with about_tab:
        st.markdown(ABOUT_TEXT, unsafe_allow_html=False)

    with viz_tab:
        st.title("📈 Volatility Surface Visualizer")

        # Symbols list
        try:
            with open("data/processed/symbols.txt") as f:
                symbols = [s.strip() for s in f]
        except:
            symbols = ["AAPL", "MSFT", "GOOGL"]

        # Sidebar inputs
        stock = st.sidebar.selectbox("Stock", symbols)
        default_date = datetime.date(2024, 1, 3)  # Known trading day
        date  = st.sidebar.date_input("Date", default_date)
        surf_type = st.sidebar.radio("Surface Type", ["Realized", "Predicted", "Both"])
        generate = st.sidebar.button("Generate")

        if generate:
            # Load and transform
            ohlcv = load_data("ohlcv", stock)
            df     = transformation_pipeline(ohlcv)
            df     = df.with_columns(pl.col("date").cast(str))
            dstr   = date.strftime("%Y-%m-%d")

            figs = {}
            if surf_type in ("Realized", "Both"):
                fr, sr = generate_vol_surface(df, stock, dstr, show_surface=False)
                figs["Realized"] = (fr, sr)
            if surf_type in ("Predicted", "Both"):
                model = load_lgbm_model()
                pdf   = predict_for_visualization(model, df, stock, dstr)
                pdf   = pdf.with_columns(pl.col("date").cast(str))
                fp, sp = generate_vol_surface(pdf, stock, dstr, show_surface=False)
                figs["Predicted"] = (fp, sp)

            # Display
            if surf_type == "Both":
                rtab, ptab = st.tabs(["Realized", "Predicted"])
                with rtab:
                    st.plotly_chart(figs["Realized"][0], use_container_width=True)
                    display_surface_details(stock, figs["Realized"][1], "Realized")
                with ptab:
                    st.plotly_chart(figs["Predicted"][0], use_container_width=True)
                    display_surface_details(stock, figs["Predicted"][1], "Predicted")
            else:
                fig, surf = figs[surf_type]
                st.plotly_chart(fig, use_container_width=True)
                display_surface_details(stock, surf, surf_type)

if __name__ == "__main__":
    main()
