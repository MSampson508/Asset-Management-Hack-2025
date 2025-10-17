"""
Produce 2-regime labels and probabilities from a monthly macro dataset.

Two public functions:
1) regime_classifier(start, end, feature_cols=None, n_regimes=2, random_state=42)
   -> Returns a dict: 
        {state_id: pandas.DatetimeIndex of dates assigned to that regime (hard labels)}
2) regime_probs(start, end, feature_cols=None, n_regimes=2, random_state=42, min_train_months=36)
   -> Returns a pandas.DataFrame indexed by Date with columns ["regime_0", ..., "regime_{K-1}"]
      giving the *filtered, past-only* probability that each month belongs to each regime.

Assumptions
- df is a pandas.DataFrame indexed by a DatetimeIndex at monthly frequency (month-end or close).
- Columns include the macro features you want to use (levels, deltas, and/or z-scores).
- No I/O in this module; upstream code loads and cleans data.

Notes
- For training segmentation, we allow using smoothed posteriors
  via regime_classifier (standard approach).
- For validation/testing, regime_probs fits the HMM on an expanding
  window and returns *filtered* (past-only) probs.
- Feature selection: if feature_cols is None, we auto-select
  columns ending in "_Z"; else all numeric.

Requirements
- numpy, pandas
- hmmlearn >= 0.2 (pip install hmmlearn)  OR conda install -c conda-forge hmmlearn
"""

import os
from typing import Optional, Dict, List
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from hmmlearn.hmm import GaussianHMM

_HAS_SCIPY = True

_REF_FP_PATH = "regime_fp_reference.csv"   # where we persist fingerprints
_SIG_FEATS_PATH = "macro_features_clean.csv"  # interpretable z-features

def _load_signature_features(sig_feats_path: str = _SIG_FEATS_PATH) -> pd.DataFrame:
    """Load interpretable z-scored macro features used for fingerprints."""
    if not os.path.exists(sig_feats_path):
        # If not available, return empty; alignment becomes no-op.
        return pd.DataFrame()
    df_sig = pd.read_csv(sig_feats_path, parse_dates=["date"]).set_index("date").sort_index()
    # keep only z columns if present
    zcols = [c for c in df_sig.columns if c.endswith("_Z")]
    if zcols:
        df_sig = df_sig[zcols]
    return df_sig

def _state_fingerprints(probs: pd.DataFrame, sig_feats: pd.DataFrame) -> pd.DataFrame:
    """
    Compute probability-weighted mean for each state over signature features.
    probs: [T x K] with columns regime_0..regime_{K-1} aligned to sig_feats.index
    returns: [K x M] DataFrame (rows = regimes)
    """
    if probs.empty or sig_feats.empty:
        return pd.DataFrame()
    # Align on intersection of dates
    idx = probs.index.intersection(sig_feats.index)
    P = probs.loc[idx]
    X = sig_feats.loc[idx]
    K = P.shape[1]
    fps = []
    for k in range(K):
        w = P.iloc[:, k].to_numpy().reshape(-1, 1)  # [T x 1]
        num = np.nansum(w * X.to_numpy(), axis=0)
        den = float(np.nansum(w))
        fps.append(num / max(den, 1e-12))
    return pd.DataFrame(fps, index=[f"regime_{k}" for k in range(K)], columns=X.columns)

def _save_reference_fp(fp: pd.DataFrame, path: str = _REF_FP_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fp.to_csv(path, index=True)

def _load_reference_fp(path: str = _REF_FP_PATH) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path, index_col=0)

def _align_mapping(fp_new: pd.DataFrame, fp_ref: pd.DataFrame) -> dict:
    """
    Return mapping {new_id -> ref_id} such that fingerprints align.
    If sizes differ, match the min(K_new, K_ref) and map remaining greedily.
    """
    if fp_new.empty or fp_ref.empty:
        return {i: i for i in range(len(fp_new))}  # identity when no reference
    # Build distance matrix
    A = fp_new.to_numpy()
    B = fp_ref.to_numpy()
    # If shapes differ, align common columns
    common_cols = fp_new.columns.intersection(fp_ref.columns)
    if len(common_cols) == 0:
        return {i: i for i in range(len(fp_new))}
    A = fp_new[common_cols].to_numpy()
    B = fp_ref[common_cols].to_numpy()

    if _HAS_SCIPY:
        D = cdist(A, B, metric="euclidean")
        rows, cols = linear_sum_assignment(D)
        mapping = {int(r): int(c) for r, c in zip(rows, cols)}
    else:
        # Greedy fallback (not optimal but works fine for K<=4)
        remaining_ref = set(range(B.shape[0]))
        mapping = {}
        for i in range(A.shape[0]):
            dists = [(j, np.linalg.norm(A[i] - B[j])) for j in remaining_ref]
            j_best = min(dists, key=lambda x: x[1])[0]
            mapping[i] = j_best
            remaining_ref.remove(j_best)

    # Ensure all states have a mapping (identity for extras)
    for i in range(fp_new.shape[0]):
        mapping.setdefault(i, i)
    return mapping

def _apply_mapping_to_probs(probs: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rename = {
        f"regime_{k}": f"regime_{mapping[k]}"
        for k in mapping
        if f"regime_{k}" in probs.columns
    }
    out = probs.rename(columns=rename)
    # Reorder columns to standard order regime_0..regime_{K-1} if all present
    cols_all = [c for c in sorted(out.columns, key=lambda c: int(c.split("_")[1]))]
    return out.reindex(columns=cols_all)

def _apply_mapping_to_labels(hard: pd.Series, mapping: dict) -> pd.Series:
    return hard.map(mapping)

df = pd.read_csv("macro_features_clean_pca90.csv", parse_dates=["date"]).set_index("date")

def _select_features(df: pd.DataFrame, feature_cols: Optional[List[str]]) -> pd.DataFrame:
    """Pick feature columns. Prefer *_Z standardized columns if feature_cols is None."""
    if feature_cols is not None and len(feature_cols) > 0:
        X = df[feature_cols].copy()
    else:
        zcols = [c for c in df.columns if c.endswith("_Z")]
        if len(zcols) > 0:
            X = df[zcols].copy()
        else:
            # Fallback to numeric columns
            X = df.select_dtypes(include=[np.number]).copy()
    # Drop columns that are entirely NaN
    X = X.dropna(axis=1, how="all")
    return X


def _fit_hmm(X: pd.DataFrame, n_regimes: int = 2, random_state: int = 42) -> GaussianHMM:
    """Fit a Gaussian HMM on features X (pandas DataFrame)."""
    # simple nan handling: forward/back fill then drop residuals
    Xf = X.copy().ffill().bfill().dropna(how="any")
    if Xf.empty:
        raise ValueError("After ffill/bfill, X is empty. Provide more complete features.")
    model = GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=500,
        tol=1e-4,
        random_state=random_state,
        verbose=False,
        init_params="stmcw",
    )
    model.fit(Xf.values)
    model._fitted_index = Xf.index  # remember which rows were used (for alignment)
    return model


def _posterior_probs(model: GaussianHMM, X: pd.DataFrame) -> pd.DataFrame:
    """Return posterior (smoothed) probabilities aligned to X's index."""
    X_aligned = X.reindex(model._fitted_index).copy().ffill().bfill().dropna(how="any")
    _, post = model.score_samples(X_aligned.values)
    probs = pd.DataFrame(
        post,
        index=X_aligned.index,
        columns=[f"regime_{i}" for i in range(model.n_components)],
    )
    return probs


def regime_classifier(
    start: str,
    end: str,
    feature_cols: Optional[List[str]] = None,
    n_regimes: int = 2,
    random_state: int = 42,
) -> Dict[int, pd.DatetimeIndex]:
    """
    Fit a K-regime Gaussian HMM on df.loc[start:end, feature_cols] and return the dates for each
    regime using hard labels (argmax posterior).

    Uses *smoothed* posteriors (common for offline regime labeling in training).

    Returns: dict[int, DatetimeIndex]
    """

    df = pd.read_csv("macro_features_clean_pca90.csv", parse_dates=["date"]).set_index("date")
    sub = df.loc[pd.to_datetime(start): pd.to_datetime(end)].copy()
    if sub.empty:
        raise ValueError("No rows in the specified date range.")
    X = _select_features(sub, feature_cols)
    model = _fit_hmm(X, n_regimes=n_regimes, random_state=random_state)

    probs = _posterior_probs(model, X)
    hard = probs.idxmax(axis=1).str.replace("regime_", "", regex=False).astype(int)

    # ---------- NEW: align to reference & persist if needed ----------
    sig_feats = _load_signature_features()
    fp_new = _state_fingerprints(probs, sig_feats)
    fp_ref = _load_reference_fp()
    if fp_ref.empty and not fp_new.empty:
        # First run: set reference from this training slice
        _save_reference_fp(fp_new)
        mapping = {i: i for i in range(n_regimes)}
    else:
        mapping = _align_mapping(fp_new, fp_ref)
        # Remap probs/labels to stable IDs
        probs = _apply_mapping_to_probs(probs, mapping)
        hard = _apply_mapping_to_labels(hard, mapping)
    # ---------------------------------------------------------------

    groups: Dict[int, pd.DatetimeIndex] = {}
    for k in range(n_regimes):
        groups[k] = hard[hard == k].index
    return groups


def regime_probs(
    start: str,
    end: str,
    feature_cols: Optional[List[str]] = None,
    n_regimes: int = 2,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute filtered (past-only) regime probabilities for each month in [start, end],
    fitting on ALL history up to each month t (not just within the [start, end] slice).
    """
    df = pd.read_csv("macro_features_clean_pca90.csv", parse_dates=["date"]).set_index("date")
    # Full features from the entire df (for training history)
    X_full = _select_features(df, feature_cols).copy()
    X_full = X_full.ffill().bfill()

    # Output index is only the requested window
    idx_out = df.loc[pd.to_datetime(start): pd.to_datetime(end)].index
    if len(idx_out) == 0:
        raise ValueError("No rows in the specified date range.")

    out = pd.DataFrame(
        index=idx_out,
        columns=[f"regime_{k}" for k in range(n_regimes)],
        dtype=float
    )

    def logsumexp_vec(a: np.ndarray) -> float:
        amax = np.max(a)
        return float(amax + np.log(np.sum(np.exp(a - amax))))

    for t in idx_out:
        # use all data up to and including t
        X_train = X_full.loc[:t].dropna(how="any")
        #X_train = X_train.iloc[:-1]

        model = _fit_hmm(X_train, n_regimes=n_regimes, random_state=random_state)

        # Forward filter to get P(state_t | x_<=t)
        log_emlik = model._compute_log_likelihood(X_train.values)  # (T, K)
        log_startprob = np.log(model.startprob_ + 1e-16)
        log_trans = np.log(model.transmat_ + 1e-16)

        T, K = log_emlik.shape
        alpha = np.empty((T, K), dtype=float)
        alpha[0] = log_startprob + log_emlik[0]
        alpha[0] -= logsumexp_vec(alpha[0])
        for tt in range(1, T):
            prev = alpha[tt - 1][:, None] + log_trans
            alpha[tt] = log_emlik[tt] + np.array([logsumexp_vec(prev[:, j]) for j in range(K)])
            alpha[tt] -= logsumexp_vec(alpha[tt])

        out.loc[t] = np.exp(alpha[-1])

    # ---------- NEW: align to saved reference ----------
    sig_feats = _load_signature_features()
    fp_new = _state_fingerprints(out.dropna(), sig_feats)
    fp_ref = _load_reference_fp()
    if not fp_ref.empty and not fp_new.empty:
        mapping = _align_mapping(fp_new, fp_ref)
        out = _apply_mapping_to_probs(out, mapping)
    # ---------------------------------------------------

    return out

# TESTING #
if __name__ == "__main__":

    # --- Quick smoke test on your CSV ---
    try:
        # Load your combined + z-scored dataset
        df = pd.read_csv("macro_features_clean_pca90.csv", parse_dates=["date"]).set_index("date")
        all_months = pd.date_range("2012-01-31", "2025-12-31", freq="M")
        all_years = pd.date_range("2012-01-31", "2025-12-31", freq="Y")
        print("Data loaded:", df.shape, "rows/cols")

        # TRAINING: get hard regime labels on early sample
        '''for year in all_years:
            groups = regime_classifier(start="2005-01-31", end=year)
            print(groups)
            for k, dates in groups.items():
                print(f"Regime {k}: {len(dates)} months")

        # VALIDATION: get probabilities for later sample
        for month in all_months:
        # Pass the same month as start and end
            probs = regime_probs(start=month, end=month)

            print(f"=== {month.strftime('%Y-%m')} ===")
            print(probs)
            print()'''
        probs = regime_probs("2015-01-31", "2025-11-30")
        print(probs["regime_0"])

    except Exception as e:
        print("Test run failed:", e)
