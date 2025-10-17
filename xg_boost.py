"""
expert_regime_training.py

This module trains and evaluates two XGBoost-based "expert" models — each specialized for 
a distinct market regime — using GPU acceleration via CUDA. It integrates with the 
`regime_seperator` module to identify regime-specific periods, trains separate experts 
on regime-filtered data, and then combines their predictions during testing.

Main functionalities:
---------------------
1. **Data Preparation**
   - Splits the dataset into training and validation (test) sets based on date ranges.
   - Filters and retains only relevant numerical features for model training.

2. **Regime Separation**
   - Uses `regime_seperator.regime_classifier()` to partition the date range into two 
     distinct regimes (e.g., bull vs. bear markets).

3. **Model Training**
   - Trains two XGBoost regressors (`expert_r1` and `expert_r2`) on GPU (via cuDF and CUDA).
   - Each expert is trained on data corresponding to its respective regime.

4. **Model Testing**
   - Combines predictions from both experts for a test dataset.
   - Optionally weights predictions by regime probabilities.

5. **Evaluation**
   - Computes an approximate R² metric for performance evaluation.
   - Designed for extendability and diagnostic experimentation.

Functions:
-----------
    initializeDataFrames(df, start, end):
        Splits data into training and testing sets by date.

    get_regime_dates(start, end):
        Retrieves regime-specific date ranges from the regime separator.

    trainExperts(df_train, r1_dates, r2_dates):
        Trains two XGBoost expert models on regime-specific subsets of the training data.

    testExperts(df_test, expert_r1, expert_r2):
        Generates predictions from both experts and assembles them into a DataFrame.

    r_squared_test(df_test, expert_r1, expert_r2):
        Computes an approximate R² performance metric for expert predictions.

    main():
        Executes a full pipeline test for all module functions using sample data.

Notes:
------
- Requires CUDA-enabled XGBoost build (`xgboost.build_info().get("USE_CUDA") == True`).
- Depends on `regime_seperator` for regime classification and probability estimation.
"""
import numpy as np
import pandas as pd
import xgboost as xgb
from xgboost import XGBRegressor
# Make sure user is using GPU with their xgboost build
assert xgb.build_info().get("USE_CUDA") == True
import regime_seperator
import cudf

#initializes df_train and df_val based on the date range - saves last year provided for validation
def initializeDataFrames(df, start, end):
    """
    Split a DataFrame into training and testing sets based on a year range.

    This function filters the input DataFrame to include only rows where the
    'ret_eom' (return end-of-month) date falls within a specified range for
    training, and assigns rows after that range to the test set. It also
    removes non-feature columns to prepare the data for model input.

    Parameters
    ----------
    df : pandas.DataFrame
        The input DataFrame containing at least the columns 'ret_eom' (datetime)
        and other numerical or categorical features used for model training.

    start : int
        The starting year (inclusive) for the training period.

    end : int
        The ending year (inclusive) for the training period. All rows with a year
        greater than `end` are assigned to the test set.

    Returns
    -------
    df_train : pandas.DataFrame
        A filtered DataFrame containing rows between `start` and `end` (inclusive),
        with non-feature columns removed.

    df_test : pandas.DataFrame
        A filtered DataFrame containing rows with years greater than `end`,
        with non-feature columns removed.

    Notes
    -----
    - The function assumes the 'ret_eom' column is of datetime type.
    - Columns such as 'date', 'iid', 'excntry', 'year', etc., are dropped before returning.
    """
    # Whatever you want to drop
    cols_to_drop = {"date", "iid", "excntry", "year", "id", "month", "char_date", "char_eom"}
    keep_cols = [c for c in df.columns if c not in cols_to_drop]

    #initialize training and validation samples
    train_years = df["ret_eom"].dt.year
    df_train = df.loc[(train_years >= start) & (train_years <= end), keep_cols].copy()
    df_test = df.loc[(train_years > end), keep_cols].copy()

    return df_train, df_test
# Should train each expert on regime specific data

def get_regime_dates(start, end):
    """
    Retrieve regime-specific date ranges within a given period.

    This function calls the `regime_classifier` from the `regime_seperator` module
    to obtain two sets of dates corresponding to distinct market regimes
    (e.g., regime 1 and regime 2) over the specified date range.

    Parameters
    ----------
    start : str or datetime-like
        The start date of the period (inclusive). Can be a string in ISO format
        (e.g., "2005-01-01") or a datetime object.

    end : str or datetime-like
        The end date of the period (inclusive). Can be a string in ISO format
        (e.g., "2005-12-31") or a datetime object.

    Returns
    -------
    r1_dates : list or pandas.Series
        A list or Series of dates corresponding to the first identified regime.

    r2_dates : list or pandas.Series
        A list or Series of dates corresponding to the second identified regime.

    Notes
    -----
    - The behavior and format of the returned dates depend on the implementation
      of `regime_seperator.regime_classifier`.
    - Typically used to separate training data for regime-specific model training.
    """
    groups = regime_seperator.regime_classifier(start, end)
    r1_dates = groups[0]
    r2_dates = groups[1]
    return r1_dates, r2_dates

#seperate regimes and train both 'experts' given the training data
def trainExperts(df_train, r1_dates, r2_dates):
    """
    Train two XGBoost expert models on regime-specific subsets of the training data.

    This function separates the input training DataFrame into two subsets 
    corresponding to distinct market regimes (regime 1 and regime 2), 
    using provided date lists. It then trains two XGBoost regressors on GPU 
    (via cuDF and XGBoost’s CUDA backend) — one for each regime.

    Parameters
    ----------
    df_train : pandas.DataFrame
        The full training dataset containing feature columns, 'ret_eom' (date), 
        'gvkey' (firm identifier), and 'stock_ret' (target variable).

    r1_dates : list or pandas.Series
        Dates corresponding to regime 1. Used to filter rows in `df_train`.

    r2_dates : list or pandas.Series
        Dates corresponding to regime 2. Used to filter rows in `df_train`.

    Returns
    -------
    expert_r1 : xgboost.XGBRegressor
        The trained XGBoost model fitted on regime 1 data.

    expert_r2 : xgboost.XGBRegressor
        The trained XGBoost model fitted on regime 2 data.

    Notes
    -----
    - Models are trained using GPU acceleration (`device='cuda'`).
    - Both models use identical hyperparameters (e.g., learning rate, depth, etc.).
    - If either regime subset is empty, that expert will not be trained.
    - The objective function currently uses squared error loss 
      (`'reg:squarederror'`), but may later be replaced by an R²-based objective.
    """
    #initialize regime specific datasets
    df_train_r1 = df_train.loc[df_train["ret_eom"].isin(r1_dates)].copy()
    df_train_r2 = df_train.loc[df_train["ret_eom"].isin(r2_dates)].copy()

    #need only data without dates or gvkeys
    #Xi is samples for regime i and Yi is labels for regime i
    X1 = df_train_r1.drop(columns=["ret_eom", "gvkey", "stock_ret"]).copy()
    Y1 = df_train_r1["stock_ret"]
    X2 = df_train_r2.drop(columns=["ret_eom", "gvkey", "stock_ret"]).copy()
    Y2 = df_train_r2["stock_ret"]



    params = dict(
        n_estimators=7500,
        max_depth=4,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=2.0,
        random_state=42,
        objective="reg:squarederror" #NEED TO CHANGE TO REAL R^2 METRIC
    )
    params.update({"tree_method": "hist", "device": "cuda"})
    params.update(dict(base_score=0.0))

    expert_r1 = XGBRegressor(**params)
    expert_r2 = XGBRegressor(**params)
    X1, Y1 = cudf.from_pandas(X1), cudf.from_pandas(Y1) #both for better gpu handeling
    X2, Y2 = cudf.from_pandas(X2), cudf.from_pandas(Y2)
    if not X1.empty:
        expert_r1.fit(X1,Y1)
    if not X2.empty:
        expert_r2.fit(X2,Y2)
    return expert_r1, expert_r2

def testExperts(df_test, expert_r1, expert_r2):
    """
    Generate and combine predictions from two regime-specific expert models.

    This function applies two trained XGBoost models (each corresponding to a 
    distinct market regime) to the test dataset, then merges their predictions 
    according to regime probabilities over time. The final output is a DataFrame 
    containing both individual and combined regime-weighted predictions.

    Parameters
    ----------
    df_test : pandas.DataFrame
        Test dataset containing features, 'ret_eom' (date), 'gvkey' (firm identifier),
        and 'stock_ret' (actual returns).

    expert_r1 : xgboost.XGBRegressor
        Trained XGBoost model corresponding to regime 1.

    expert_r2 : xgboost.XGBRegressor
        Trained XGBoost model corresponding to regime 2.

    Returns
    -------
    merged_df : pandas.DataFrame
        A DataFrame containing:
        - 'date': Observation dates
        - 'gvkey': Firm identifiers
        - 'r_hat1', 'r_hat2': Individual expert predictions
        - 'regime_0', 'regime_1': Regime probabilities
        - 'r_hat': Combined prediction (weighted average using regime probabilities)

    Notes
    -----
    - The regime probabilities are retrieved via `regime_seperator.regime_probs()`
      over the test period and merged on matching dates.
    - Predictions are computed on GPU using cuDF for improved performance.
    - The final regime-weighted prediction is computed as:
          r_hat = r_hat1 * P(regime_0) + r_hat2 * P(regime_1)
    - If some dates in `df_test` are not found in the regime probabilities, 
      those rows may contain NaNs after the merge.
    """
    keys = df_test["gvkey"].to_numpy()
    dates = df_test["ret_eom"].to_numpy()

    regime_probabilities = regime_seperator.regime_probs(
        df_test["ret_eom"].iloc[0],
        df_test["ret_eom"].iloc[-1]
    )

    # p1 = regime_probabilities["regime_0"].to_numpy()
    # p2 = regime_probabilities["regime_1"].to_numpy()

    X1 = df_test.drop(columns=["ret_eom", "gvkey", "stock_ret"]).copy()
    X1 = cudf.from_pandas(X1)

    r_hat1 = expert_r1.predict(X1)
    r_hat2 = expert_r2.predict(X1)
    r_hat1 = np.asarray(r_hat1)
    r_hat2 = np.asarray(r_hat2)

    df = pd.DataFrame({
    "date": dates,
    "gvkey": keys,
    "r_hat1": r_hat1,
    "r_hat2": r_hat2,
    })

    df["date"] = pd.to_datetime(df["date"])
    regime_probabilities.index = pd.to_datetime(regime_probabilities.index)
    merged_df = df.merge(regime_probabilities, left_on="date", right_index=True, how="left")

    merged_df["r_hat"] = (
        merged_df["r_hat1"] * merged_df["regime_0"]
        + merged_df["r_hat2"] * merged_df["regime_1"]
    )

    return merged_df

def r_squared_test(df_test, expert_r1, expert_r2):
    """
    Compute the R² (coefficient of determination) score for the expert ensemble.

    This function evaluates the performance of the combined regime-based prediction
    generated by `testExperts()` against the actual stock returns in the test dataset.
    It calculates the R² metric, which measures how well the model explains the 
    variance in the observed data.

    Parameters
    ----------
    df_test : pandas.DataFrame
        Test dataset containing the actual returns ('stock_ret') and the features
        used for prediction.

    expert_r1 : xgboost.XGBRegressor
        Trained XGBoost model corresponding to regime 1.

    expert_r2 : xgboost.XGBRegressor
        Trained XGBoost model corresponding to regime 2.

    Returns
    -------
    float
        The R² score, computed as:
            R² = 1 - Σ(r - r̂)² / Σ(r)²
        where `r` are actual returns and `r̂` are predicted returns.

    Notes
    -----
    - Calls `testExperts()` internally to generate combined regime-weighted predictions.
    - A higher R² value (closer to 1) indicates better model performance.
    - This implementation uses a manual R² computation rather than 
      `sklearn.metrics.r2_score` for simplicity.
    """
    r_hat = testExperts(df_test, expert_r1, expert_r2)["r_hat"].to_numpy()
    r = df_test["stock_ret"].to_numpy()
    r_squared = 1 - (np.sum(np.square(r-r_hat))/np.sum(np.square(r)))
    return r_squared

# testing

from sklearn.metrics import r2_score

# def evaluate_model(df_train, df_test, expert_r1, expert_r2):
#     """Compute and print R² scores on train and validation sets."""
#     # --- Training performance ---
#     y_train = df_train["stock_ret"].to_numpy()
#     y_train_pred = testExperts(df_train, expert_r1, expert_r2)[1]
#     train_r2 = r2_score(y_train, y_train_pred)

#     # --- Validation performance ---
#     y_test = df_test["stock_ret"].to_numpy()
#     y_test_pred = testExperts(df_test, expert_r1, expert_r2)[1]
#     val_r2 = r2_score(y_test, y_test_pred)

#     print("\n--- Model Performance ---")
#     print(f"Train R²:       {train_r2:.4f}")
#     print(f"Validation R²:  {val_r2:.4f}")
#     return train_r2, val_r2


def main():
    """Main entry point to rigorously test all functions in this module."""

    filename = r"/teamspace/studios/this_studio/IntermediateTrainingDataPickle.pkl"
    df = pd.read_pickle(filename)
    df = df.head(1000)
    input(df.head())

    # ---------------------------------------------------------
    # 2. Initialize train/test dataframes
    # ---------------------------------------------------------
    print("\n[2] Initializing data splits...")
    df_train, df_test = initializeDataFrames(df, 2005, 2005)
    print(f"Train size: {df_train.shape}, Test size: {df_test.shape}")
    input(df_train.head())

    # ---------------------------------------------------------
    # 3. Get regime date groups
    # ---------------------------------------------------------
    print("\n[3] Getting regime date groups...")
    r1_dates, r2_dates = get_regime_dates("2005-01-01", "2005-12-31")
    print(f"Regime 1 dates: {len(r1_dates)} entries, Regime 2 dates: {len(r2_dates)} entries")

    # ---------------------------------------------------------
    # 4. Train experts
    # ---------------------------------------------------------
    print("\n[4] Training regime experts...")
    expert_r1, expert_r2 = trainExperts(df_train, r1_dates, r2_dates)
    print("✓ Experts trained successfully")

    # ---------------------------------------------------------
    # 5. Test experts
    # ---------------------------------------------------------
    print("\n[5] Testing experts on validation data...")
    keys, preds = testExperts(df_test, expert_r1, expert_r2)
    print(f"Predictions generated: {len(preds)} values")
    print(f"Sample predictions: {preds[:5]}")

    # ---------------------------------------------------------
    # 6. Compute R² test metric
    # ---------------------------------------------------------
    print("\n[6] Evaluating R² performance...")
    r2_val = r_squared_test(df_test, expert_r1, expert_r2)
    print(f"Validation R² (approx): {r2_val:.4f}")

    print("\n✅ All functions executed successfully on synthetic data.")




if __name__ == "__main__":
    main()
