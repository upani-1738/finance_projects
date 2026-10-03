# Symbolic-Regression Alpha Discovery with VaR-Constrained Portfolio Construction

A quantitative research pipeline that uses genetic programming to discover readable stock-selection formulas, combines the ones that hold up out of sample into a long/short portfolio under a hard Value-at-Risk limit, and tests itself to make sure the backtest isn't fooling its author.

## Overview

Genetic programming evolves cross-sectional formulas over risk, volatility, momentum, liquidity and options-pricing features for 50 large-cap U.S. stocks. Survivors are screened on held-out data, decorrelated, stripped of generic factor exposure, and converted to expected returns. A conic optimizer then builds a dollar-neutral portfolio that cannot exceed its VaR budget, tightening that budget automatically when the VIX is elevated.

```
maximize   mu'w - (gamma/2) w'Sigma w - kappa ||w - w_prev||_1
subject to z_0.95 * sqrt(w'Sigma w) <= VaR budget
```

The VaR limit is a second-order cone constraint, solved exactly with a conic solver. Transaction costs sit inside the objective.

## Pipeline

1. **Features.** Eighteen strictly backward-looking signals: historical VaR and expected shortfall, volatility structure, momentum and reversal, market beta, liquidity, and the Black-Scholes cost of a protective put. All are normalized cross-sectionally each day; VIX enters as a regime variable.
2. **Search.** Tree-based genetic programming finds one formula for the whole cross-section. Fitness is information ratio minus penalties for turnover and formula size.
3. **Selection.** Candidates must clear a held-out validation slice, near-duplicates are dropped, and the rest are equal-weighted.
4. **Alpha scaling.** The blended signal is orthogonalized against beta, momentum, volatility and size, then scaled with Grinold's rule (IC x volatility x score) with a capped IC and shrinkage toward zero.
5. **Risk and portfolio.** Ledoit-Wolf shrinkage covariance, the VaR-constrained optimizer, volatility targeting, and spread plus square-root market-impact costs.
6. **Backtest.** Purged and embargoed walk-forward testing so overlapping labels never leak future information into training.
7. **Evaluation.** Net-of-cost performance, Probabilistic and Deflated Sharpe ratios that account for every formula tried, and Kupiec and Christoffersen tests of the VaR model.

## Validation

The `--self-test` mode checks that:

- on a synthetic panel with **zero alpha**, the pipeline finds nothing;
- on a synthetic panel with a **planted signal**, it recovers it out of sample;
- labels contain **no lookahead**;
- the optimizer's output **respects its VaR budget**.

## Usage

```bash
pip install -r requirements.txt

python qsr_system.py --self-test         # harness validation
python qsr_system.py --synthetic-null    # should find ~nothing
python qsr_system.py --synthetic-alpha   # should find the planted signal
python qsr_system.py --real              # 50 S&P names + VIX, 2018-2024
```

Outputs go to `output/`: equity curve, weights, discovered formulas, a JSON summary, and performance and monthly-return charts.

## Disclaimer

Research code for educational purposes. Not investment advice.
