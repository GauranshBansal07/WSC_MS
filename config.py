"""
Configuration for IC diagnostic checks.

Universe: Nifty 50 (fixed current composition — acceptable for diagnostic
purposes since we're testing IC properties, not precise alpha attribution).

# Lookback windows match the paper: 1, 6, 12, 36, 60 months.
"""

HISTORICAL_COMPOSITION_CSV = 'data/historical_composition.csv'
NIFTY_NEXT_50_COMPOSITION_CSV = 'data/nifty_next_50_composition.csv'

# Momentum formation periods (months) — matches the paper
LOOKBACK_WINDOWS = [1, 6, 12, 36, 60]

# Data range: go back far enough for the 60-month lookback to have
# a meaningful IC time series. 60 months of lookback + ~10 years of
# IC observations means we need data from ~2010.
DATA_START = '2008-01-01'
DATA_END = '2025-04-01'

# Transaction cost assumptions: 10 bps per side (20 bps round-trip)
TRANSACTION_COST_BPS = 10

# Leverage cost for vol-scaled gross exposure > 1.0 (Nifty futures roll cost proxy)
# Barroso-Santa-Clara 2015 implementation — Indian market adjustment
LEVERAGE_COST_ANNUAL = 0.05    # 5% annualized (primary spec)

# Risk-free rate proxy (India 10Y govt bond, approximate)
RISK_FREE_ANNUAL = 0.07
