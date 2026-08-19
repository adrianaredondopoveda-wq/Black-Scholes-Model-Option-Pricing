"""Production-style option-pricing engine for European vanilla options.

All public pricing methods accept scalars or NumPy arrays, making the engine
convenient for both a single trade and an entire volatility surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

OptionType = Literal["call", "put"]


@dataclass(frozen=True)
class Option:
    spot: float
    strike: float
    maturity: float  # years
    rate: float
    volatility: float
    kind: OptionType = "call"
    dividend_yield: float = 0.0

    def validate(self) -> None:
        if self.spot <= 0 or self.strike <= 0:
            raise ValueError("spot and strike must be positive")
        if self.maturity <= 0 or self.volatility <= 0:
            raise ValueError("maturity and volatility must be positive")
        if self.kind not in ("call", "put"):
            raise ValueError("kind must be 'call' or 'put'")


@dataclass(frozen=True)
class Greeks:
    delta: float
    gamma: float
    vega: float       # change in price for a +1 volatility point
    theta: float      # daily time decay
    rho: float        # change in price for a +1 interest-rate point


def _d1_d2(spot, strike, maturity, rate, volatility, dividend_yield=0.0):
    spot, strike, maturity, volatility = map(np.asarray, (spot, strike, maturity, volatility))
    root_t = np.sqrt(maturity)
    d1 = (np.log(spot / strike) + (rate - dividend_yield + 0.5 * volatility**2) * maturity) / (volatility * root_t)
    return d1, d1 - volatility * root_t


def price(spot, strike, maturity, rate, volatility, kind: OptionType = "call", dividend_yield=0.0):
    """Black-Scholes-Merton price with continuous dividend yield."""
    if kind not in ("call", "put"):
        raise ValueError("kind must be 'call' or 'put'")
    d1, d2 = _d1_d2(spot, strike, maturity, rate, volatility, dividend_yield)
    pv_spot = np.asarray(spot) * np.exp(-dividend_yield * maturity)
    pv_strike = np.asarray(strike) * np.exp(-rate * maturity)
    if kind == "call":
        return pv_spot * norm.cdf(d1) - pv_strike * norm.cdf(d2)
    return pv_strike * norm.cdf(-d2) - pv_spot * norm.cdf(-d1)


def greeks(option: Option) -> Greeks:
    option.validate()
    d1, d2 = _d1_d2(option.spot, option.strike, option.maturity, option.rate, option.volatility, option.dividend_yield)
    q, t, s, k, r, sigma = option.dividend_yield, option.maturity, option.spot, option.strike, option.rate, option.volatility
    carry = np.exp(-q * t)
    pdf = norm.pdf(d1)
    gamma = carry * pdf / (s * sigma * np.sqrt(t))
    vega = s * carry * pdf * np.sqrt(t) / 100
    if option.kind == "call":
        delta = carry * norm.cdf(d1)
        theta_year = -(s * carry * pdf * sigma) / (2 * np.sqrt(t)) - r * k * np.exp(-r * t) * norm.cdf(d2) + q * s * carry * norm.cdf(d1)
        rho = k * t * np.exp(-r * t) * norm.cdf(d2) / 100
    else:
        delta = carry * (norm.cdf(d1) - 1)
        theta_year = -(s * carry * pdf * sigma) / (2 * np.sqrt(t)) + r * k * np.exp(-r * t) * norm.cdf(-d2) - q * s * carry * norm.cdf(-d1)
        rho = -k * t * np.exp(-r * t) * norm.cdf(-d2) / 100
    return Greeks(float(delta), float(gamma), float(vega), float(theta_year / 365), float(rho))


def no_arbitrage_bounds(spot: float, strike: float, maturity: float, rate: float, kind: OptionType, dividend_yield=0.0) -> tuple[float, float]:
    pv_s = spot * np.exp(-dividend_yield * maturity)
    pv_k = strike * np.exp(-rate * maturity)
    return (max(0.0, pv_s - pv_k), pv_s) if kind == "call" else (max(0.0, pv_k - pv_s), pv_k)


def implied_volatility(market_price: float, spot: float, strike: float, maturity: float, rate: float, kind: OptionType = "call", dividend_yield=0.0) -> float:
    """Robust Brent solver; rejects prices outside theoretical bounds."""
    lower, upper = no_arbitrage_bounds(spot, strike, maturity, rate, kind, dividend_yield)
    if not lower - 1e-10 <= market_price <= upper + 1e-10:
        raise ValueError(f"market price outside no-arbitrage bounds [{lower:.4f}, {upper:.4f}]")
    objective = lambda sigma: price(spot, strike, maturity, rate, sigma, kind, dividend_yield) - market_price
    return float(brentq(objective, 1e-6, 8.0, xtol=1e-10))


def monte_carlo_price(option: Option, paths: int = 100_000, seed: int = 7) -> tuple[float, float]:
    """Risk-neutral GBM estimator returning (price, 95% CI half-width)."""
    option.validate()
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(paths // 2)
    z = np.concatenate([z, -z])  # antithetic variates reduce estimator variance
    terminal = option.spot * np.exp((option.rate - option.dividend_yield - 0.5 * option.volatility**2) * option.maturity + option.volatility * np.sqrt(option.maturity) * z)
    payoff = np.maximum(terminal - option.strike, 0) if option.kind == "call" else np.maximum(option.strike - terminal, 0)
    discounted = np.exp(-option.rate * option.maturity) * payoff
    return float(discounted.mean()), float(1.96 * discounted.std(ddof=1) / np.sqrt(len(discounted)))
