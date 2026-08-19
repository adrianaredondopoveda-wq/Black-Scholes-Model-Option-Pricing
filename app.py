import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.engine import Option, greeks, implied_volatility, monte_carlo_price, price

st.set_page_config(page_title="QuantLab | Black-Scholes", page_icon="◈", layout="wide")
st.title("◈ QuantLab — Black-Scholes-Merton Workbench")
st.caption("European options · continuous dividends · Greeks · implied volatility · Monte Carlo validation")

with st.sidebar:
    st.header("Trade inputs")
    kind = st.selectbox("Option type", ["call", "put"])
    spot = st.number_input("Spot price", min_value=0.01, value=100.0)
    strike = st.number_input("Strike", min_value=0.01, value=105.0)
    days = st.number_input("Days to expiry", min_value=1, value=180)
    volatility = st.slider("Volatility", 0.01, 2.0, 0.25, 0.01)
    rate = st.slider("Risk-free rate", -0.05, 0.20, 0.035, 0.001)
    dividend = st.slider("Dividend yield", 0.0, 0.15, 0.0, 0.001)

t = days / 365
option = Option(spot, strike, t, rate, volatility, kind, dividend)
bs_price = float(price(spot, strike, t, rate, volatility, kind, dividend))
g = greeks(option)
mc, ci = monte_carlo_price(option)

a, b, c = st.columns(3)
a.metric("Model price", f"${bs_price:,.4f}")
b.metric("Monte Carlo", f"${mc:,.4f}", f"95% CI ± ${ci:.4f}")
c.metric("Moneyness", f"{spot / strike:.3f}×")

st.subheader("Risk sensitivities")
metrics = {"Delta": g.delta, "Gamma": g.gamma, "Vega / 1 vol pt": g.vega, "Theta / day": g.theta, "Rho / 1 rate pt": g.rho}
st.dataframe(pd.DataFrame(metrics.items(), columns=["Greek", "Value"]).set_index("Greek").style.format("{:.6f}"), use_container_width=True)

left, right = st.columns(2)
spots = np.linspace(spot * 0.55, spot * 1.45, 120)
with left:
    st.subheader("Price profile")
    fig = go.Figure(go.Scatter(x=spots, y=price(spots, strike, t, rate, volatility, kind, dividend), line=dict(color="#58D68D", width=3)))
    fig.add_vline(x=spot, line_dash="dash", line_color="#AAB7B8")
    fig.update_layout(xaxis_title="Underlying spot", yaxis_title="Option value", template="plotly_dark", height=360, margin=dict(l=10,r=10,t=20,b=10))
    st.plotly_chart(fig, use_container_width=True)
with right:
    st.subheader("Volatility surface")
    grid_spot = np.linspace(spot * .70, spot * 1.30, 40)
    grid_vol = np.linspace(max(.02, volatility - .20), volatility + .20, 35)
    xx, yy = np.meshgrid(grid_spot, grid_vol)
    zz = price(xx, strike, t, rate, yy, kind, dividend)
    fig = go.Figure(go.Surface(x=xx, y=yy, z=zz, colorscale="Viridis"))
    fig.update_layout(scene=dict(xaxis_title="Spot", yaxis_title="Volatility", zaxis_title="Price"), template="plotly_dark", height=360, margin=dict(l=0,r=0,t=20,b=0))
    st.plotly_chart(fig, use_container_width=True)

st.subheader("Implied-volatility desk")
market = st.number_input("Observed market option price", min_value=0.0, value=float(round(bs_price, 4)), step=0.01)
try:
    iv = implied_volatility(market, spot, strike, t, rate, kind, dividend)
    st.success(f"Implied volatility: **{iv:.2%}**  |  Difference vs. input: **{(iv - volatility):+.2%}**")
except ValueError as error:
    st.error(str(error))
