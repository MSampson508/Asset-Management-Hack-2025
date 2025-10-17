"""
Regime separator (stable, warm-started HMM) — DEFAULT: 2 REGIMES

Public API (unchanged names):
- regime_classifier(start, end, feature_cols=None, n_regimes=2, random_state=42)
    Returns a dict mapping {state_id: DatetimeIndex of assigned months} using *smoothed*
    posteriors on the in-sample window [start, end].

- regime_probs(start, end, feature_cols=None, n_regimes=2, random_state=42,
               min_train_months=36)
    Returns a DataFrame indexed by month end in [start, end] with columns
    ["regime_0", "regime_1"] (for K=2 by default) giving *filtered (past-only)* probabilities.

Design goals:
- Stable: enforce a minimum expanding-history window and diagonal covariances
- Fast: warm-start monthly re-fits from the previous month’s parameters
- Consistent labeling across runs: optional fingerprint alignment to a saved
  reference (based on interpretable z-scored macro features when available)

Expected data files (default paths below; adjust as needed):
- macro_features_clean_pca90.csv  (features; must have a 'date' column)
- macro_features_clean.csv        (optional signature features with *_Z columns)
- regime_fp_reference.csv         (optional saved reference fingerprints)

Dependencies: numpy, pandas, hmmlearn, scipy (for Hungarian assignment)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional
import os
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from scipy.optimize import linear_sum_assignment

# ------------------------
# Configurable file paths
# ------------------------
_FEATURES_PATH = "macro_features_clean_pca90.csv"
_SIG_FEATS_PATH = "macro_features_clean.csv"        # for fingerprints (interpretable *_Z)
_REF_FP_PATH   = "regime_fp_reference.csv"          # persistent reference fingerprints

# ------------------------
# Utilities
# ------------------------

def _month_index(start: pd.Timestamp | str, end: pd.Timestamp | str) -> pd.DatetimeIndex:
    s = pd.to_datetime(start)
    e = pd.to_datetime(end)
    if s > e:
        raise ValueError("start must be <= end")
    return pd.date_range(s, e, freq="M")


def _select_features(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> pd.DataFrame:
    if feature_cols is not None and len(feature_cols) > 0:
        X = df[feature_cols].copy()
    else:
        # Prefer *_Z columns if present, else all numeric
        zcols = [c for c in df.columns if c.endswith("_Z")]
        X = df[zcols].copy() if zcols else df.select_dtypes(include=[np.number]).copy()
    # Basic cleaning
    X = X.replace([np.inf, -np.inf], np.nan).ffill().bfill()
    return X


def _load_features(path: str = _FEATURES_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Features file not found: {path}")
    df = pd.read_csv(path, parse_dates=["date"]).set_index("date").sort_index()
    # Normalize to month-end stamps to avoid index-mismatch (e.g., 2025-10-31 not found)
    try:
        df.index = df.index.to_period("M").to_timestamp("M")
    except Exception:
        pass
    return df


def _load_signature_features(sig_feats_path: str = _SIG_FEATS_PATH) -> pd.DataFrame:
    """Load interpretable z-scored macro features used for fingerprints (optional)."""
    if not os.path.exists(sig_feats_path):
        return pd.DataFrame()
    df_sig = pd.read_csv(sig_feats_path, parse_dates=["date"]).set_index("date").sort_index()
    zcols = [c for c in df_sig.columns if c.endswith("_Z")]
    return df_sig[zcols].copy() if zcols else pd.DataFrame()


# ------------------------
# HMM fitting & filtering
# ------------------------

def _fit_hmm(
    X: pd.DataFrame,
    n_regimes: int = 2,
    random_state: int = 42,
    *,
    n_iter: int = 200,
    tol: float = 1e-3,
    covariance_type: str = "diag",
    init_params: str = "stmcw",
    model: Optional[GaussianHMM] = None,
) -> GaussianHMM:
    """Fit a GaussianHMM.

    If `model` provided, it will be *warm-started* by setting `init_params=""` and
    re-using its current parameters.
    """
    Xv = X.values.astype(float)
    if model is None:
        model = GaussianHMM(
            n_components=n_regimes,
            covariance_type=covariance_type,
            n_iter=n_iter,
            tol=tol,
            random_state=random_state,
            init_params=init_params,
            verbose=False,
        )
    else:
        # Warm-start: keep current parameters
        model.n_iter = n_iter
        model.tol = tol
        model.init_params = ""  # do not re-init
        model.covariance_type = covariance_type

    # Convergence warnings are common but often benign; suppress noise
    with warnings.catch_warnings():
        try:
            from sklearn.exceptions import ConvergenceWarning
            warnings.simplefilter("ignore", category=ConvergenceWarning)
        except Exception:
            pass
        model.fit(Xv)

    # stash fitted index for later reference
    model._fitted_index = X.index  # type: ignore[attr-defined]
    return model


def _forward_filtered_probs(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Return filtered (past-only) probabilities for each row in X.

    hmmlearn exposes `predict_proba`, but that is the *smoothed* posterior.
    Here we compute normalized forward messages (alpha) to get filtered probs.
    """
    Xv = X.values.astype(float)
    log_emlik = model._compute_log_likelihood(Xv)  # (T, K)
    log_start = np.log(model.startprob_ + 1e-16)
    log_trans = np.log(model.transmat_ + 1e-16)

    T, K = log_emlik.shape
    alpha = np.empty((T, K), dtype=float)

    def _lse(vec: np.ndarray) -> float:
        m = float(np.max(vec))
        return m + float(np.log(np.sum(np.exp(vec - m))))

    # init
    alpha[0] = log_start + log_emlik[0]
    alpha[0] -= _lse(alpha[0])

    # forward recursion
    for t in range(1, T):
        tmp = alpha[t - 1][:, None] + log_trans  # (K, K)
        alpha[t] = log_emlik[t] + np.array([_lse(tmp[:, j]) for j in range(K)])
        alpha[t] -= _lse(alpha[t])

    probs = np.exp(alpha)
    cols = [f"regime_{k}" for k in range(K)]
    return pd.DataFrame(probs, index=X.index, columns=cols)


# ------------------------
# Fingerprint alignment (optional but helpful for consistent labels)
# ------------------------

def _state_fingerprints(probs: pd.DataFrame, sig_feats: pd.DataFrame) -> pd.DataFrame:
    """Probability-weighted mean of signature features per regime.

    probs: DataFrame with columns `regime_0..regime_{K-1}`
    sig_feats: DataFrame indexed by date with interpretable *_Z columns
    """
    if probs.empty or sig_feats.empty:
        return pd.DataFrame()

    # Align to overlapping dates
    common = probs.index.intersection(sig_feats.index)
    if len(common) == 0:
        return pd.DataFrame()

    P = probs.loc[common]
    S = sig_feats.loc[common]

    Ks = [c for c in P.columns if c.startswith("regime_")]
    out = {}
    for k, col in enumerate(Ks):
        w = P[col].values.reshape(-1, 1)
        num = (w * S.values).sum(axis=0)
        den = w.sum()
        out[k] = num / max(den, 1e-12)
    fp = pd.DataFrame(out).T
    fp.index.name = "regime"
    fp.columns = list(S.columns)
    return fp


def _save_reference_fp(fp: pd.DataFrame, path: str = _REF_FP_PATH) -> None:
    if fp.empty:
        return
    fp.to_csv(path, index=True)


def _load_reference_fp(path: str = _REF_FP_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)


def _align_mapping(fp_new: pd.DataFrame, fp_ref: pd.DataFrame) -> Dict[int, int]:
    """Return a mapping {new_state -> ref_state} via Hungarian algorithm on cosine distance."""
    if fp_new.empty or fp_ref.empty:
        return {}
    # match on common columns
    common_cols = list(sorted(set(fp_new.columns).intersection(fp_ref.columns)))
    if not common_cols:
        return {}
    A = fp_new[common_cols].values
    B = fp_ref[common_cols].values

    # cosine distance (robust to scale)
    A_norm = A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B_norm = B / (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    D = 1.0 - A_norm @ B_norm.T  # cosine distance matrix

    r, c = linear_sum_assignment(D)
    return {int(r_i): int(c_i) for r_i, c_i in zip(r, c)}


def _apply_mapping_to_probs(probs: pd.DataFrame, mapping: Dict[int, int]) -> pd.DataFrame:
    if not mapping:
        return probs
    cols = [c for c in probs.columns if c.startswith("regime_")]
    K = len(cols)
    new_cols = {f"regime_{k}": f"regime_{mapping.get(k, k)}" for k in range(K)}
    out = probs.rename(columns=new_cols)
    # Ensure column order regime_0..K-1 exists
    missing = set(cols) - set(out.columns)
    for m in missing:
        out[m] = 0.0
    return out[[f"regime_{k}" for k in range(K)]]


def _apply_mapping_to_labels(labels: pd.Series, mapping: Dict[int, int]) -> pd.Series:
    if not mapping:
        return labels
    return labels.map(lambda x: mapping.get(int(x), int(x)))


# ------------------------
# Public API — DEFAULT n_regimes=2
# ------------------------

def regime_classifier(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    feature_cols: Optional[List[str]] = None,
    n_regimes: int = 2,
    random_state: int = 42,
) -> Dict[int, pd.DatetimeIndex]:
    """Fit on [start, end] and return hard labels (via smoothed posteriors) per regime.

    This is for in-sample segmentation. We use smoothed posteriors because they
    generally yield better regime assignments in-sample.
    """
    df = _load_features(_FEATURES_PATH)
    # Use intersection to avoid KeyError if requested months extend past data range
    req_idx = _month_index(start, end)
    window_idx = df.index.intersection(req_idx)
    window = df.loc[window_idx]
    if window.empty:
        raise ValueError("No rows in the specified date range.")

    X = _select_features(window, feature_cols)
    if X.isna().any().any():
        X = X.ffill().bfill().dropna(how="any")
    if X.empty:
        raise ValueError("Selected features are empty after cleaning.")

    model = _fit_hmm(X, n_regimes=n_regimes, random_state=random_state,
                     n_iter=300, tol=1e-3, covariance_type="diag")

    # Smoothed posteriors (hmmlearn predict_proba)
    gamma = pd.DataFrame(model.predict_proba(X.values), index=X.index,
                         columns=[f"regime_{k}" for k in range(n_regimes)])

    # Optional: align to saved reference for stability
    sig = _load_signature_features(_SIG_FEATS_PATH)
    fp_new = _state_fingerprints(gamma, sig)
    fp_ref = _load_reference_fp(_REF_FP_PATH)
    mapping = _align_mapping(fp_new, fp_ref) if not fp_ref.empty else {}

    # Hard labels
    labels = gamma.idxmax(axis=1).str.replace("regime_", "", regex=False).astype(int)
    labels = _apply_mapping_to_labels(labels, mapping)

    # Save reference if none exists (bootstrap)
    if fp_ref.empty and not fp_new.empty:
        _save_reference_fp(fp_new, _REF_FP_PATH)

    out: Dict[int, pd.DatetimeIndex] = {}
    for k in range(n_regimes):
        idx = labels.index[labels == k]
        out[k] = pd.DatetimeIndex(idx)
    return out


def regime_probs(
    start: str | pd.Timestamp,
    end: str | pd.Timestamp,
    feature_cols: Optional[List[str]] = None,
    n_regimes: int = 2,
    random_state: int = 42,
    min_train_months: int = 36,
) -> pd.DataFrame:
    """Compute *filtered* regime probabilities for each month in [start, end].

    Uses an expanding history up to month t, warm-starting parameters from t-1.
    Enforces a minimum history window for stability and uses diagonal covariances.
    After computing probabilities, performs optional alignment to a saved reference
    fingerprint based on interpretable z-features, for consistent labeling.
    """
    df_full = _load_features(_FEATURES_PATH)
    X_full = _select_features(df_full, feature_cols)

    # Build the output index via intersection (robust if request extends beyond data)
    req_idx = _month_index(start, end)
    idx_out = df_full.index.intersection(req_idx)
    if len(idx_out) == 0:
        raise ValueError("No rows in the specified date range.")

    cols = [f"regime_{k}" for k in range(n_regimes)]
    out = pd.DataFrame(index=idx_out, columns=cols, dtype=float)

    model: Optional[GaussianHMM] = None
    for t in idx_out:
        X_train = X_full.loc[:t].dropna(how="any")
        if X_train.shape[0] < max(min_train_months, n_regimes * 10):
            out.loc[t] = np.nan
            continue

        model = _fit_hmm(
            X_train,
            n_regimes=n_regimes,
            random_state=random_state,
            n_iter=150 if model is not None else 250,
            tol=1e-3,
            covariance_type="diag",
            init_params="stmcw" if model is None else "",
            model=model,
        )

        # Filtered probs up to t, take last row
        filt = _forward_filtered_probs(model, X_train)
        out.loc[t, cols] = filt.iloc[-1].values

    # Alignment to reference
    sig = _load_signature_features(_SIG_FEATS_PATH)
    fp_new = _state_fingerprints(out.dropna(), sig)
    fp_ref = _load_reference_fp(_REF_FP_PATH)
    if not fp_ref.empty and not fp_new.empty:
        mapping = _align_mapping(fp_new, fp_ref)
        out = _apply_mapping_to_probs(out, mapping)
    elif fp_ref.empty and not fp_new.empty:
        # bootstrap a reference on first successful run
        _save_reference_fp(fp_new, _REF_FP_PATH)

    return out


# ------------------------
# Module smoke test (optional)
# ------------------------
if __name__ == "__main__":
    try:
        # quick sanity checks
        print("Loading features…")
        _ = _load_features(_FEATURES_PATH)
        print("OK.")
        rp = regime_probs("2015-01-31", "2016-12-31", n_regimes=2)
        print(rp.head())
    except Exception as e:
        print("Self-test failed:", e)
