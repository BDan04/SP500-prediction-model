# A Regime-Aware Macro Overlay for Equity Risk Allocation

Can cross-asset macro signals improve **5-day S&P 500 (SPY) timing**?

This project builds a machine-learning signal that ranks upcoming 5-day windows for SPY using rates, volatility term structure, currencies, commodities and global equity indices. It then tests whether that signal is useful as a tactical overlay on an equity position. Every result is measured **out of sample** using time-series cross-validation and expanding-window walk-forward testing.

## Key Findings

- **The signal has modest but real predictive power.** Expanding walk-forward IC is **0.089** and stitched out-of-sample IC is **0.063**, positive in every cross-validation fold.
- **It is strongest when markets are volatile.** IC rises from 0.034 in the lowest-volatility quartile to 0.084 in the highest.
- **The predictions separate returns.** The top decile of predictions averaged +0.50% over the next 5 days, compared with +0.05% for the bottom decile.
- **The overlay improves timing, not total return.** The volatility-targeted overlay reaches a Sharpe of about **0.85** while invested roughly 76% of the time. It trails buy-and-hold on total return (5.1x vs 7.3x) because it sits out part of a strong bull market.

## Data

| Source | Series |
|---|---|
| Yahoo Finance (`yfinance`) | SPY, ES=F, NQ=F, ^IXIC, ^DJI, ^VIX, ^VIX9D, DX-Y.NYB, ^TNX, ^FVX, ^IRX, GC=F, CL=F |
| FRED | GDP, CPI |

- **Sample:** January 2005 to September 2026 (5,441 daily observations)
- **Features (68):** levels, 1/5/10-day returns, 10/20-day volatility, drawdowns, momentum, yield-curve slope, VIX term structure, and growth, inflation and volatility regime flags
- **Target:** SPY forward 5-day return
- **Feature pruning:** permutation importance on out-of-sample IC reduced 68 features to **58**

Data is downloaded when the notebook runs. No raw data is stored in this repository.

## Methodology

**Models**
| Model | Role |
|---|---|
| Ridge Regression | Linear baseline |
| HistGradientBoostingRegressor | Main model, captures nonlinear macro interactions |
| Logistic Regression / HGB Classifier | Direction (up/down) benchmarks |
| LSTM (20-day lookback) | Deep-learning robustness check |

**Validation**
- `TimeSeriesSplit` (6 folds) with out-of-sample predictions stitched together across folds
- Expanding-window walk-forward: minimum 3 years of training data, retrained every 5 days
- Spearman IC, rolling IC, regime-conditional IC and decile spreads

**Backtest**
- Non-overlapping 5-day holding periods, so each prediction is used once with no look-ahead
- Long SPY when the prediction exceeds a threshold, otherwise in cash
- 5 bps transaction costs per trade
- Threshold sweep (zero, mean, quantiles from 50% to 80%)
- Volatility-targeted version (15% annual target)

## Results

### Predictive power (out of sample)
| Metric | Value |
|---|---|
| Expanding walk-forward IC (HGBR, pruned features) | **0.089** |
| Stitched OOS IC, HGBR (68 / 58 features) | 0.061 / 0.063 |
| Stitched OOS IC, Ridge | 0.058 |
| Rolling IC, mean | 0.081 |
| LSTM walk-forward IC | 0.057 |
| Direction classifiers (AUC) | ~0.50 to 0.51 |

The tree model edges out both the linear baseline and the LSTM. Classifiers that predict only up or down have no edge. The signal is useful for **ranking** expected returns, not for calling direction.

### IC by volatility regime (SPY 20-day volatility quartile)
| Quartile | Lowest | 2 | 3 | Highest |
|---|---|---|---|---|
| IC | 0.034 | 0.055 | 0.071 | 0.084 |

### Overlay strategy (non-overlapping 5-day trades, 5 bps costs)
| Strategy | Sharpe | Total Return | Avg. Exposure |
|---|---|---|---|
| Buy & Hold SPY | — | 7.34x | 100% |
| Overlay, best threshold rule (pruned) | 0.74 | 4.92x | 63% |
| Overlay, volatility-targeted (15%) | **0.85** | 5.11x | 76% |

### Most important features
Top drivers by out-of-sample permutation importance: **5-year yield (FVX)**, **SPY 20-day momentum**, **VIX9D**, **SPY 20-day drawdown**, **oil**, **10-year yield changes** and **S&P futures returns**. No single feature dominates, which supports using a nonlinear model.

## Limitations
- An IC below 0.10 is typical for short-horizon equity prediction, but it leaves a thin margin after costs.
- The overlay does not beat buy-and-hold on total return over 2005 to 2026, a period with a strong upward trend.
- GDP and CPI are aligned by observation date, not release date. GDP for a quarter is published about a month after the quarter ends, so the macro features contain a small look-ahead. Pruning removed the CPI features, but the GDP-based features remain in the model.
- VIX9D history starts in 2011, so early folds train without VIX term-structure features.
- Results come from one asset (SPY) and one horizon (5 days).

## Next Steps
- Combine Ridge and HGBR in an ensemble
- Test on a frozen holdout period that is never used in model development
- Add full risk diagnostics (max drawdown, VaR/CVaR, turnover) at an equal-volatility comparison with buy-and-hold

## Repository Structure
```
├── README.md
├── requirements.txt
├── notebooks/
│   └── capstone_analysis.ipynb
└── presentation/
    └── Capstone_Final.pdf
```

## How to Run
```bash
git clone https://github.com/<your-username>/<repo-name>.git
cd <repo-name>
pip install -r requirements.txt
jupyter notebook notebooks/capstone_analysis.ipynb
```
Run all cells from top to bottom. Data downloads automatically. The LSTM walk-forward cell takes about 80 minutes on a CPU. You can skip it to reproduce every other result.

**Requirements:** Python 3.10+, pandas, numpy, scikit-learn, scipy, matplotlib, yfinance, pandas-datareader, fredapi, tensorflow

## Reproducibility
- **Frozen data window:** market and macro data run from 2005-01-01 through **2026-09-29**, the date the results in this README were produced. To run on the latest data, set `END = None` and the FRED `end` to `dt.datetime.today()` in the data cell.
- **Fixed random seeds:** scikit-learn models use `random_state=42`. The LSTM uses `tf.random.set_seed(42)` and `np.random.seed(42)`.
- **Expect small differences:** Yahoo Finance recalculates historical adjusted prices after each dividend, and FRED revises past GDP and CPI values. Re-running the notebook should produce results very close to those above, but not always identical. TensorFlow results can also vary slightly by hardware.

## Author
**Brandon Daniels** · Capstone Project · 2026

*For research and educational purposes only. Not investment advice.*
