# QuantLab — Black-Scholes-Merton Workbench

An interview-ready quantitative-finance project rather than a one-function pricing script.

## What it demonstrates

- Closed-form Black-Scholes-Merton pricing for European calls and puts, including continuous dividends.
- Analytic Greeks with market-standard units: vega/rho per 1 percentage-point move and theta per day.
- A bounded, no-arbitrage-aware implied-volatility solver (Brent's method).
- Independent risk-neutral Monte Carlo validation with antithetic variates and a 95% confidence interval.
- Vectorised NumPy engine powering price profiles and an interactive price/volatility surface.
- Unit tests for a textbook result, put-call parity, IV round-tripping, Monte Carlo consistency and Greek sanity.

## Run it

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

Then open the local URL Streamlit prints. Run the quality checks with:

```bash
pytest -q
```

## Interview talking points

The analytic engine is used as a control variate: Monte Carlo is purposely separate, so the dashboard demonstrates model validation rather than simply reusing the same calculation. The IV solver validates no-arbitrage bounds before root-finding; this prevents apparently successful but economically impossible calibrations.

## Assumptions and limits

The model prices **European** vanilla options under lognormal, constant volatility and constant rates. It is not appropriate for American exercise, stochastic volatility, jumps, transaction costs, or a market volatility smile. Those limitations are intentional discussion points for a next iteration (binomial/American pricer and Heston calibration).
