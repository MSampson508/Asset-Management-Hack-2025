"""
Implements a rolling-window backtesting framework for an equity portfolio 
tilted based on an XGBoost forecasting strategy.

This module executes a sequential financial simulation using a regime-switching 
or expert-based forecasting model (imported from xg_boost). The process 
involves:
1. Preparing historical stock return data ('TrainingDataPickle.pkl') and 
   market index data ('mkt_ind.csv').
2. Performing an annual, rolling retraining of the XGBoost expert models.
3. Calculating monthly long-short portfolio returns based on the models' 
   forward predictions.
4. Computing the portfolio's realized rolling Alpha and Beta (using OLS) 
   against the market excess return (Factor Model Analysis).

The primary entry point is `tilting_portfolio()`, which returns the detailed 
trade log and the rolling alpha log.
"""

from __future__ import annotations
from typing import Tuple, List
import numpy as np
import pandas as pd

import xg_boost

def _to_year_month(s) -> pd.PeriodIndex:
    """
    Converts a pandas Series of datetime-like objects to a monthly period index.

    This is a helper function used for aligning stock data (df) returns with 
    monthly market index data (mkt).

    Args:
        s (pd.Series): A pandas Series containing datetime objects or datetime-like strings.

    Returns:
        pd.PeriodIndex: A monthly period index (e.g., '2015-01') representing 
                        the year and month of each date.
    """
    return pd.to_datetime(s).dt.to_period("M")

def _ols_alpha_beta(y: np.ndarray, x: np.ndarray) -> Tuple[float, float]:
    """
    Performs Ordinary Least Squares (OLS) regression to calculate Alpha and Beta 
    based on the Capital Asset Pricing Model (CAPM).

    The regression models the portfolio's excess return (y) against the 
    market's excess return (x), using the equation: 
    $y = \\alpha + \\beta x + \\epsilon$

    Args:
        y (np.ndarray): Dependent variable (e.g., portfolio excess returns).
        x (np.ndarray): Independent variable (e.g., market excess returns).

    Returns:
        Tuple[float, float]: The calculated alpha (intercept) and beta (slope).
    """
    X = np.column_stack([np.ones(len(x)), np.asarray(x)])
    a, b = np.linalg.lstsq(X, np.asarray(y), rcond=None)[0]
    return float(a), float(b)

def tilting_portfolio() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Executes the main rolling-window backtest simulation for the XGBoost strategy.

    The backtest runs annually, re-training the XGBoost expert models each year
    (from 2013-01 onward) and then simulating monthly long-short trading on the 
    out-of-sample data. It calculates the resulting long-short returns and 
    the corresponding rolling CAPM alpha and beta.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: 
            - trades_df (pd.DataFrame): The detailed log of monthly trade results, 
              including returns, portfolio sizes, and the training year.
            - alpha_df (pd.DataFrame): The log of rolling alpha and beta values 
              calculated each month.
    """
    df = pd.read_pickle("TrainingDataPickle.pkl").copy()
    df["ret_eom"] = pd.to_datetime(df["ret_eom"])
    df["ym"] = _to_year_month(df["ret_eom"])

    mkt = pd.read_csv("mkt_ind.csv")

    mkt["ym"] = pd.PeriodIndex(
        pd.to_datetime(
            mkt["year"].astype(int).astype(str) + "-" + mkt["month"].astype(int).astype(str) + "-01"
        ),
        freq="M",
    )

    first_train_end = pd.Period("2012-12", freq="M")
    end_limit = pd.Period("2025-12", freq="M")

    trade_logs: List[dict] = []
    alpha_logs: List[dict] = []
    pred_logs: List[pd.DataFrame] = []   # NEW: r_hat cross-section each month

    for train_end_year in range(int(first_train_end.year), int(end_limit.year) + 1):
        train_end = pd.Period(f"{train_end_year}-12", freq="M")
        model_df = df.drop(columns=["ym"], errors="ignore")
        df_train, df_test = xg_boost.initializeDataFrames(model_df, start=2005, end=int(train_end.year))
        r1_dates, r2_dates = xg_boost.get_regime_dates(f"2005-01-01", f"{train_end.year}-12-31")
        expert_r1, expert_r2 = xg_boost.trainExperts(df_train, r1_dates, r2_dates)

        _next_month = pd.period_range(start=train_end, periods=2, freq="M")[1]

        # exactly the next 12 months after training (one-year OOS window),
        # but do not go past the global end_limit
        one_year_window = pd.period_range(start=_next_month, periods=12, freq="M")
        test_months = one_year_window[one_year_window <= end_limit]
        if test_months.size == 0:
            continue

        df_test["ret_eom"] = pd.to_datetime(df_test["ret_eom"])
        df_test["ym"] = _to_year_month(df_test["ret_eom"])
        months_df = pd.DataFrame({"ym": test_months})
        df_test = df_test.merge(months_df, on="ym", how="inner").copy()

        for ym in test_months:
            month_df = df_test.loc[df_test["ym"] == ym].copy()
            if month_df.empty:
                continue

            month_df_model = month_df.drop(columns=["ym"], errors="ignore")
            preds_df = xg_boost.testExperts(month_df_model, expert_r1, expert_r2)
            preds_df = preds_df.sort_values("r_hat", ascending=False)
            
            longs = preds_df.head(100)["gvkey"].tolist()
            shorts = preds_df.tail(100)["gvkey"].tolist()

            # ----- record per-stock predictions for this month -----
            preds_store = preds_df.loc[:, ["gvkey", "r_hat"]].copy()
            preds_store["ym"] = ym
            pred_logs.append(preds_store)

            # ----- compute monthly R^2 using the same features passed to the experts -----
            r2 = xg_boost.r_squared_test(month_df_model, expert_r1, expert_r2)

            realized = month_df[["gvkey","stock_ret"]].copy()
            long_mean = realized.loc[realized["gvkey"].isin(longs), "stock_ret"].mean()
            short_mean = realized.loc[realized["gvkey"].isin(shorts), "stock_ret"].mean()
            long_short_ret = float(long_mean - short_mean)

            trade_logs.append({
                "ym": ym, "long_mean": long_mean, "short_mean": short_mean,
                "long_short_return": long_short_ret, "n_long": len(longs), "n_short": len(shorts),
                "train_end_year": train_end.year, "r_squared": r2
            })
            preds_df_all = (
                pd.concat(pred_logs, axis=0, ignore_index=True)
                    .loc[:, ["ym", "gvkey", "r_hat"]]
                    .sort_values(["ym", "r_hat"], ascending=[True, False])
                    .reset_index(drop=True)
            )
            perf_df = pd.DataFrame(trade_logs).drop_duplicates("ym", keep="last").sort_values("ym")
            merged = perf_df.merge(mkt, on="ym", how="inner")
            merged["port_excess"] = merged["long_short_return"] - merged["rf"]
            merged["mkt_excess"]  = merged["ret"] - merged["rf"]

            if len(merged) >= 3:
                a, b = _ols_alpha_beta(
                    merged["port_excess"].to_numpy(),
                    merged["mkt_excess"].to_numpy()
                    )
                alpha_logs.append({"ym": ym, "alpha": a, "beta": b})
                print("one window complete")
            else:
                alpha_logs.append({"ym": ym, "alpha": np.nan, "beta": np.nan})
                print("moving to next window")

    trades_df = pd.DataFrame(trade_logs).sort_values(["ym","train_end_year"]).reset_index(drop=True)
    alpha_df  = pd.DataFrame(alpha_logs).drop_duplicates("ym", keep="last").sort_values("ym").reset_index(drop=True)
    return trades_df, alpha_df, preds_df_all

def alpha_performance(trades_df: pd.DataFrame) -> Tuple[float, float]:
    """
    Calculates the overall, unconditional CAPM Alpha and Beta for the portfolio.

    This function merges the portfolio's monthly long-short returns with 
    market data ('mkt_ind.csv') to derive both portfolio and market excess 
    returns. It then uses OLS regression via `_ols_alpha_beta` across 
    the full period to determine the realized Alpha (risk-adjusted return) 
    and Beta (market risk exposure).

    Args:
        trades_df (pd.DataFrame): DataFrame containing the portfolio's 
                                  monthly trading results, including 
                                  'long_short_return' and 'ym'.

    Returns:
        Tuple[float, float]: A tuple containing the portfolio's 
                             (unconditional Alpha, unconditional Beta).
    """
    mkt = pd.read_csv("mkt_ind.csv")

    mkt["ym"] = pd.PeriodIndex(
        pd.to_datetime(
            mkt["year"].astype(int).astype(str) + "-" + mkt["month"].astype(int).astype(str) + "-01"
        ),
        freq="M",
    )

    port = trades_df.drop_duplicates("ym")
    merged = port.merge(mkt, on="ym", how="inner")
    merged["port_excess"] = merged["long_short_return"] - merged["rf"]
    merged["mkt_excess"]  = merged["ret"] - merged["rf"]

    a, b = _ols_alpha_beta(merged["port_excess"].to_numpy(), merged["mkt_excess"].to_numpy())
    return a, b

def main():
    trades_df, alpha_df, preds_df = tilting_portfolio()
    trades_df.to_csv("/teamspace/studios/this_studio/trades.csv", index=False)
    alpha_df.to_csv("/teamspace/studios/this_studio/alpha.csv", index=False)
    preds_df.to_csv("/teamspace/studios/this_studio/preds.csv", index=False)


if __name__ == "__main__":
    main()

