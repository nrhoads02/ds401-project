import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime

from src.data_transformation.transformation_pipeline import transformation_pipeline
from src.data_extraction.dataframe_loader import load_data
from src.data_modeling.lgbm_modeling import load_lgbm_model, predict_for_visualization


def generate_vol_surface(df, stock, date, show_surface=False):
    """
    Generate a volatility surface with proper calendar day time horizons 
    and standardized moneyness.
    
    Parameters:
    -----------
    df : pl.DataFrame
        DataFrame containing _future columns
    stock : str
        Stock symbol
    date : str or datetime.date or pl.Date or pl.Expr
        Date for visualization
    show_surface : bool
        Whether to display the surface (default: False)
    
    Returns:
    --------
    tuple: (fig, surface_dict)
        fig: plotly figure object
        surface_dict: dictionary with surface data
    """    
    # Filter data for the specific stock and date
    row = df.filter((pl.col("act_symbol") == stock) & (pl.col("date") == date))
    
    if row.height == 0:
        raise ValueError(f"No data found for {stock} on {date}")
    
    # Define trading day windows
    trading_windows = [10, 15, 20, 25, 30, 35]
    # Convert trading days to approximate calendar days
    calendar_days = [int(round(window * 7/5)) for window in trading_windows]
    
    # Extract realized values and surface parameters
    realized_data = {}
    for i, trading_window in enumerate(trading_windows):
        days = calendar_days[i]
        years = days / 365.0
        
        realized_vol = float(row[f"YZVol_{trading_window}_future"][0]) * np.sqrt(252)
        realized_k   = float(row[f"LogPriceRatio_{trading_window}_future"][0])
        skew         = float(row[f"VolSkew_{trading_window}_future"][0])
        curvature    = float(row[f"VolCurvature_{trading_window}_future"][0])
        wing_ratio   = float(row[f"WingRatio_{trading_window}_future"][0])
        mean_rev     = float(row[f"MeanReversion_{trading_window}_future"][0])
        vol_of_vol   = float(row[f"VolOfVol_{trading_window}_future"][0])
        price_vol_corr = float(row[f"PriceVolCorr_{trading_window}_future"][0])
        vol_intensity  = float(row[f"VolIntensity_{trading_window}_future"][0])
        
        realized_data[days] = {
            'vol': realized_vol,
            'moneyness': realized_k,
            'skew': skew,
            'curvature': curvature,
            'wing_ratio': wing_ratio,
            'mean_reversion': mean_rev,
            'vol_of_vol': vol_of_vol,
            'price_vol_corr': price_vol_corr,
            'vol_intensity': vol_intensity,
            'years': years
        }
    
    # Determine moneyness grid
    kms = [d['moneyness'] for d in realized_data.values()]
    m_min, m_max = min(kms) - 0.15, max(kms) + 0.15
    moneyness = np.linspace(m_min, m_max, 50)
    
    # Build meshgrid
    K, T = np.meshgrid(moneyness, calendar_days)
    vol_surface = np.zeros_like(K)
    
    # Populate surface
    for i, days in enumerate(calendar_days):
        data = realized_data[days]
        rv, k0, yrs = data['vol'], data['moneyness'], data['years']
        sqrt_yr = np.sqrt(yrs)
        for j, k in enumerate(moneyness):
            rel = k - k0
            rel_s = rel / (rv * sqrt_yr)
            bf = min(yrs*5, 0.8)
            eff_k = (1-bf)*rel + bf*rel_s
            # components
            skew_eff  = (data['skew'] * abs(eff_k) * data['wing_ratio']
                         if rel<0 else -data['skew']*eff_k/data['wing_ratio'])
            curv_eff  = data['curvature'] * eff_k**2 * (1+2*data['vol_of_vol'])
            corr_eff  = data['price_vol_corr'] * eff_k * 0.5
            inten_eff = (data['vol_intensity']-0.5)*2*abs(eff_k)**2
            vol_lvl   = 0.05*data['vol_of_vol']*yrs
            comb      = skew_eff + curv_eff + corr_eff + inten_eff
            v = rv * (1+comb)*(1+vol_lvl)
            vol_surface[i,j] = np.clip(v, 0.05, 2.0)
    
    # Plotly figure
    fig = go.Figure()
    fig.add_trace(go.Surface(x=K, y=T, z=vol_surface,
                             colorscale='Viridis',
                             colorbar=dict(title='Ann. Vol'),
                             name='Surface'))
    # Realized points
    actual_m = [realized_data[d]['moneyness'] for d in calendar_days]
    actual_v = [realized_data[d]['vol'] for d in calendar_days]
    fig.add_trace(go.Scatter3d(x=actual_m, y=calendar_days, z=actual_v,
                               mode='markers',
                               marker=dict(color='red', size=6),
                               name='Realized'))
    fig.update_layout(
        title=f"Vol Surface for {stock} on {date}",
        scene=dict(
            xaxis_title='Log Moneyness',
            yaxis_title='Cal. Days',
            zaxis_title='Ann. Vol'
        ),
        width=900, height=700
    )
    if show_surface:
        fig.show()
    
    surface_data = {
        'K': K.tolist(),
        'T': T.tolist(),
        'vol_surface': vol_surface.tolist(),
        'calendar_days': calendar_days,
        'moneyness': moneyness.tolist(),
        'actual_moneyness': actual_m,
        'actual_vols': actual_v,
        'date': str(date),
        'stock': stock
    }
    return fig, surface_data


def main():
    st.set_page_config(
        page_title="Stock Volatility Surface Visualizer",
        page_icon="📈",
        layout="wide"
    )
    
    st.title("📊 Stock Volatility Surface Visualizer")
    st.write("Select a stock and date to visualize the volatility surface")
    
    # Sidebar inputs
    try:
        with open("data/processed/symbols.txt", "r") as f:
            symbols = [s.strip() for s in f]
    except:
        symbols = ["AAPL","MSFT","GOOGL"]
    
    with st.sidebar:
        st.header("Parameters")
        stock = st.selectbox("Select Stock:", symbols, index=0)
        date = st.date_input("Select Date:", value=datetime.date.today())
        surface_type = st.radio("Surface Type:", ["Realized","Predicted","Both"])
        st.markdown("---")
        st.markdown("### About")
        st.markdown("""
**What is a Local Volatility Surface?**

A *volatility surface* shows how the market’s expectation of future volatility varies across option strike prices and times to expiration. Classic models assume constant volatility, but real market prices imply a “smile” or “skew” in volatility depending on moneyness (ratio of strike to spot) and maturity.

A *local volatility surface* σ(S,t) is the instantaneous volatility the market would assign to an underlying price S at time t, calibrated so that model prices match observed option prices across strikes and maturities.

**What is a Realized Local Volatility Surface?**

Instead of using option prices (implied volatility), a *realized* local volatility surface is built from actual historical price movements. It maps how volatility truly behaved in the past for each combination of price level and time horizon. By comparing this realized surface to the theoretical local volatility implied by the options market, we can identify potential mispricings or inefficiencies.
        """)
        st.markdown("---")
        submit = st.button("Generate")

    if submit:
        try:
            ohlcv_df = load_data("ohlcv", stock)
            transformed = transformation_pipeline(ohlcv_df)
            
            # Handle date lookup
            date_str = date.strftime("%Y-%m-%d")
            df_str = transformed.with_columns(pl.col("date").cast(str))
            available = df_str["date"].unique().to_list()
            if date_str not in available:
                closest = min(available, key=lambda d: abs((datetime.datetime.strptime(d, "%Y-%m-%d").date() - date).days))
                st.warning(f"No data for {date_str}, using nearest {closest}.")
                date_str = closest
            
            surfaces = {}
            if surface_type in ["Realized","Both"]:
                fig_r, surf_r = generate_vol_surface(df_str, stock, date_str)
                surfaces["Realized"] = (fig_r, surf_r)
            if surface_type in ["Predicted","Both"]:
                model = load_lgbm_model()
                pred_df = predict_for_visualization(model, transformed, stock, date_str)
                pred_df = pred_df.with_columns(pl.col("date").cast(str))
                fig_p, surf_p = generate_vol_surface(pred_df, stock, date_str)
                surfaces["Predicted"] = (fig_p, surf_p)
            
            if surface_type == "Both":
                tab1, tab2 = st.tabs(["Realized","Predicted"])
                with tab1:
                    st.plotly_chart(surfaces["Realized"][0], use_container_width=True)
                with tab2:
                    st.plotly_chart(surfaces["Predicted"][0], use_container_width=True)
            else:
                st.plotly_chart(surfaces[surface_type][0], use_container_width=True)
        
        except Exception as e:
            st.error(f"Error: {e}")
            import traceback; st.error(traceback.format_exc())


if __name__ == "__main__":
    main()
