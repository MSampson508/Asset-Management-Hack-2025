"""Just testing the xh_boost code in a new file"""
import pandas as pd
import cudf
import xg_boost as xg

try:
    import cudf.pandas
    cudf.pandas.install()
    print("cuDF pandas accelerator mode enabled.")
except ImportError:
    print("cuDF not available, falling back to CPU pandas.")


def main():
    """
    Main function to test and validate the XGBoost regime expert models.

    Workflow:
    1. Load a subset of the intermediate training dataset (first 300,000 rows).
    2. Initialize train and test dataframes based on specified year ranges.
    3. Retrieve regime-specific date groups for training.
    4. Train regime-specific expert models on the training data.
    5. Test the experts on a subset of the training data and generate predictions.
    6. Compute an approximate R² metric for validation performance.
    7. Optionally, display a full evaluation report.

    Notes:
    - The predictions step may need adjustment as noted in testExperts.
    - Designed for large datasets; uses subsets for testing to reduce memory usage.
    - Includes interactive `input()` statements for stepwise inspection.
    """
    filename = r"/teamspace/studios/this_studio/IntermediateTrainingDataPickle.pkl"
    df = pd.read_pickle(filename)
   

    # ---------------------------------------------------------
    # 2. Initialize train/test dataframes
    # ---------------------------------------------------------
    print("\n[2] Initializing data splits...")
    df_train, df_test = xg.initializeDataFrames(df, 2005, 2010)
    print(df_train.tail())
    input(df_test.head())
    print(f"Train size: {df_train.shape}, Test size: {df_test.shape}")

    # ---------------------------------------------------------
    # 3. Get regime date groups
    # ---------------------------------------------------------
    print("\n[3] Getting regime date groups...")
    r1_dates, r2_dates = xg.get_regime_dates("2005-01-01", "2010-12-31")
    print(f"Regime 1 dates: {len(r1_dates)} entries, Regime 2 dates: {len(r2_dates)} entries")

    # ---------------------------------------------------------
    # 4. Train experts
    # ---------------------------------------------------------
    print("\n[4] Training regime experts...")
    expert_r1, expert_r2 = xg.trainExperts(df_train, r1_dates, r2_dates)
    input("✓ Experts trained successfully")

    # ---------------------------------------------------------
    # 5. Test experts
    # ---------------------------------------------------------

    # Everything works until this point but the prediction code is wrong
    # at the comment in testExperts
    print("\n[5] Testing experts on validation data...")
    r_hat = xg.testExperts(df_test.head(50000), expert_r1, expert_r2)
    print(f"Predictions generated: {len(r_hat)} values")
    print("Sample predictions:")
    print(r_hat["r_hat"])

    # ---------------------------------------------------------
    # 6. Compute R² test metric
    # ---------------------------------------------------------
    print("\n[6] Evaluating R² performance...")
    r2_val = xg.r_squared_test(df_test.head(50000), expert_r1, expert_r2)
    print(f"Validation R² (approx): {r2_val:.4f}")

    # ---------------------------------------------------------
    # 7. Use evaluation wrapper
    # ---------------------------------------------------------
    print("\n[7] Full evaluation report:")

    print("\n✅ All functions executed successfully on synthetic data.")




if __name__ == "__main__":
    main()
