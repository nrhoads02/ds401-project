import os
import sys
# Add the project root directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import plotly.graph_objects as go
import numpy as np
import polars as pl
import datetime

# Now import should work
from src.data_transformation.transformation_pipeline import transformation_pipeline
from src.data_extraction.dataframe_loader import load_data


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
    
    # Define trading day windows (these are the windows in your _future columns)
    trading_windows = [10, 15, 20, 25, 30, 35]
    
    # Convert trading days to approximate calendar days
    # A common approximation: calendar days ≈ trading days × (7/5)
    calendar_days = [int(round(window * 7/5)) for window in trading_windows]
    
    # Extract realized values for all timepoints
    realized_data = {}
    for i, trading_window in enumerate(trading_windows):
        days = calendar_days[i]
        years = days / 365.0  # For term structure effects
        
        # Extract base realized values
        realized_vol = float(row[f"YZVol_{trading_window}_future"][0]) * np.sqrt(252)  # Annualize
        realized_k = float(row[f"LogPriceRatio_{trading_window}_future"][0])
        
        # Extract surface parameter values with proper bounds
        skew = float(row[f"VolSkew_{trading_window}_future"][0])
        curvature = min(max(float(row[f"VolCurvature_{trading_window}_future"][0]), -5.0), 5.0)  # Bound curvature
        
        # Ensure wing_ratio is bounded away from zero
        wing_ratio = max(float(row[f"WingRatio_{trading_window}_future"][0]), 0.2)
        
        # Bound mean reversion to reasonable values
        mean_reversion = min(max(float(row[f"MeanReversion_{trading_window}_future"][0]), 0.01), 10.0)
        
        # Ensure vol_of_vol is positive
        vol_of_vol = max(float(row[f"VolOfVol_{trading_window}_future"][0]), 0.01)
        
        # Ensure price_vol_corr is within [-1, 1]
        price_vol_corr = min(max(float(row[f"PriceVolCorr_{trading_window}_future"][0]), -1.0), 1.0)
        
        # Ensure vol_intensity is within [0, 1]
        vol_intensity = min(max(float(row[f"VolIntensity_{trading_window}_future"][0]), 0.0), 1.0)
        
        realized_data[days] = {
            'vol': realized_vol,
            'moneyness': realized_k,
            'skew': skew,
            'curvature': curvature,
            'wing_ratio': wing_ratio,
            'mean_reversion': mean_reversion,
            'vol_of_vol': vol_of_vol,
            'price_vol_corr': price_vol_corr,
            'vol_intensity': vol_intensity,
            'trading_window': trading_window,
            'years': years
        }
    
    # Determine moneyness range dynamically
    realized_moneyness = [realized_data[t]['moneyness'] for t in calendar_days]
    min_realized = min(realized_moneyness)
    max_realized = max(realized_moneyness)
    
    # Create wider range around realized moneyness points
    moneyness_padding = 0.15
    moneyness_min = min_realized - moneyness_padding
    moneyness_max = max_realized + moneyness_padding
    moneyness = np.linspace(moneyness_min, moneyness_max, 50)
    
    # Create meshgrid for surface plotting using calendar days
    K, T = np.meshgrid(moneyness, calendar_days)
    vol_surface = np.zeros(K.shape)
    
    # Generate surface with correct mathematics
    for i, days in enumerate(calendar_days):
        data = realized_data[days]
        realized_vol = data['vol']
        realized_k = data['moneyness']
        years = data['years']
        
        # Square root of time factor for standardization
        sqrt_years = np.sqrt(years)
        
        for j, k in enumerate(moneyness):
            # Calculate relative moneyness (distance from realized moneyness)
            relative_k = k - realized_k
            
            # IMPROVEMENT 1: Standardize relative moneyness by volatility and time
            # This creates more consistent shapes across maturities
            relative_k_scaled = relative_k / (realized_vol * sqrt_years)
            
            # Use a blend of raw and scaled moneyness for better behavior
            # (100% scaled can be too extreme for short dates with high vol)
            blend_factor = min(years * 5, 0.8)  # More scaling for longer horizons
            effective_k = (1 - blend_factor) * relative_k + blend_factor * relative_k_scaled
            
            # Skew component - asymmetric effect
            skew_effect = 0
            if relative_k < 0:  # Put side (typically higher volatility)
                skew_effect = data['skew'] * abs(effective_k) * data['wing_ratio']
            else:  # Call side
                skew_effect = -data['skew'] * effective_k / data['wing_ratio']
            
            # Curvature component - creates the smile/frown
            curvature_effect = data['curvature'] * effective_k**2 * (1 + 2.0 * data['vol_of_vol'])
            
            # Price-vol correlation effect
            corr_effect = data['price_vol_corr'] * effective_k * 0.5
            
            # Intensity effect - affects tails
            intensity_effect = (data['vol_intensity'] - 0.5) * 2.0 * abs(effective_k)**2
            
            # IMPROVEMENT 2: Add vol_of_vol effect on total level
            # Many stochastic vol models have vol-of-vol affect total variance
            vol_level_effect = 0.05 * data['vol_of_vol'] * years
            
            # Combined effect - must equal 0 when k = realized_k
            combined_effect = skew_effect + curvature_effect + corr_effect + intensity_effect
            
            # Final volatility calculation
            vol = realized_vol * (1.0 + combined_effect) * (1.0 + vol_level_effect)
            
            # Ensure volatility stays within reasonable bounds
            vol = max(0.05, min(vol, 2.0))
            vol_surface[i, j] = vol
    
    # Create 3D surface plot
    fig = go.Figure()
    
    # Add main surface
    fig.add_trace(go.Surface(
        x=K, 
        y=T, 
        z=vol_surface, 
        colorscale='Viridis',
        colorbar=dict(title='Annualized Volatility'),
        name='Volatility Surface'
    ))
    
    # Add actual realized points
    actual_moneyness = [realized_data[t]['moneyness'] for t in calendar_days]
    actual_vols = [realized_data[t]['vol'] for t in calendar_days]
    
    fig.add_trace(go.Scatter3d(
        x=actual_moneyness,
        y=calendar_days,
        z=actual_vols,
        mode='markers',
        marker=dict(size=7, color='red'),
        name='Realized Volatility'
    ))
    
    # Add text annotation showing trading days to calendar days mapping
    annotations = []
    for i, (trade_days, cal_days) in enumerate(zip(trading_windows, calendar_days)):
        annotations.append(
            dict(
                showarrow=False,
                x=min_realized,
                y=cal_days,
                z=0,
                text=f"{trade_days}td",
                xanchor="left",
                font=dict(color="white", size=10)
            )
        )
    
    # Properly format the date for the title
    if isinstance(date, pl.Date) or isinstance(date, pl.Expr):
        # If it's a Polars Date or Expression, convert to string in YYYY-MM-DD format
        date_value = row["date"][0]
        if hasattr(date_value, 'strftime'):
            date_str = date_value.strftime("%Y-%m-%d")
        else:
            date_str = str(date_value).split(' ')[0]  # Take just the date part if there's a time component
    elif isinstance(date, datetime.date):
        # If it's a Python datetime.date object
        date_str = date.strftime("%Y-%m-%d")
    else:
        # If it's already a string or something else
        date_str = str(date).split(' ')[0]  # Take just the date part if there's a time component
    
    # Set axis labels and title with properly formatted date
    fig.update_layout(
        title=f"Realized Volatility Surface for {stock} on {date_str}",
        scene=dict(
            xaxis_title='Log(Future Price / Spot Price) ~ log-moneyness',
            yaxis_title='Calendar Days',
            zaxis_title='Annualized Volatility',
            xaxis=dict(range=[moneyness_min, moneyness_max]),
            yaxis=dict(range=[min(calendar_days), max(calendar_days)]),
            zaxis=dict(range=[0, np.nanmax(vol_surface) * 1.1]),
            annotations=annotations
        ),
        width=900,
        height=700,
        margin=dict(r=20, l=10, b=10, t=50)
    )
    
    # Pack surface data using calendar days
    surface_data = {
        'K': K.tolist(),  # Convert numpy arrays to lists for JSON serialization
        'T': T.tolist(),
        'vol_surface': vol_surface.tolist(),
        'calendar_days': calendar_days,
        'trading_windows': trading_windows,
        'moneyness': moneyness.tolist(),
        'actual_moneyness': actual_moneyness,
        'actual_vols': actual_vols,
        'realized_data': realized_data,
        'date': date_str,  # Use the properly formatted date
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
    
    # Read ticker symbols from file
    try:
        with open("data/processed/symbols.txt", "r") as f:
            symbols = [line.strip() for line in f.readlines()]
    except FileNotFoundError:
        st.error("Symbols file not found. Make sure 'data/processed/symbols.txt' exists.")
        symbols = ["AAPL", "MSFT", "GOOGL"]  # Fallback symbols
    
    # Create sidebar for inputs
    with st.sidebar:
        st.header("Parameters")
        
        # Create a stock selector with search functionality
        stock = st.selectbox(
            "Select Stock:",
            options=symbols,
            index=symbols.index("AAPL") if "AAPL" in symbols else 0,
            help="Choose a stock ticker to visualize"
        )
        
        # Date selector
        default_date = datetime.date(2022, 5, 6)  # Default from example
        date = st.date_input(
            "Select Date:",
            value=default_date,
            help="Choose a date to visualize the volatility surface"
        )
        
        # Submit button
        submit_button = st.button("Generate Volatility Surface", use_container_width=True)
        
        # Add information about the app
        st.markdown("---")
        st.markdown("### About")
        st.info(
            "This app visualizes volatility surfaces for stocks based on "
            "realized volatility metrics and surface parameters."
        )
    
    # Main content area
    if submit_button:
        try:
            # Convert date to polars Date format
            pl_date = pl.date(date.year, date.month, date.day)
            
            # Show loading spinner while processing
            with st.spinner(f"Loading and processing data for {stock}..."):
                # Load and filter data for the selected stock (FIX: Use the selected stock, not hardcoded "AAPL")
                ohlcv_df = load_data("ohlcv", stock)
                
                # Apply the transformation pipeline
                ohlcv_df = transformation_pipeline(ohlcv_df)
            
            # Generate volatility surface
            with st.spinner("Generating volatility surface..."):
                fig, surface = generate_vol_surface(ohlcv_df, stock, pl_date, show_surface=False)
                
                # Display the figure using Streamlit's Plotly integration
                st.plotly_chart(fig, use_container_width=True)
            
            # Display some surface metadata
            with st.expander("Surface Details", expanded=False):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("Stock Information")
                    st.write(f"**Stock:** {stock}")
                    st.write(f"**Date:** {surface['date']}")  # Use the properly formatted date
                    
                    # Display some basic stats
                    if surface['actual_vols']:
                        avg_vol = sum(surface['actual_vols']) / len(surface['actual_vols'])
                        st.write(f"**Average Volatility:** {avg_vol:.2f}%")
                
                with col2:
                    st.subheader("Trading to Calendar Days")
                    # Show trading windows and corresponding calendar days
                    mapping_data = {f"{t} trading days": f"{c} calendar days" 
                                   for t, c in zip(surface['trading_windows'], surface['calendar_days'])}
                    st.json(mapping_data)
            
        except ValueError as ve:
            st.error(f"Error: {str(ve)}")
            if "No data found" in str(ve):
                st.info("Try selecting a different date or stock.")
                
        except Exception as e:
            st.error(f"An unexpected error occurred: {str(e)}")
            st.info("Please check your data files and try again.")

if __name__ == "__main__":
    main()