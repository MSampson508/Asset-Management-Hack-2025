"""
xg_boost_main.py
================

Main script for managing data preprocessing and integration with the XGBoost-based
model training pipeline.

This module performs the following:
- Enables GPU acceleration for pandas via cuDF (if available).
- Loads and preprocesses financial return data.
- Merges textual probability features (FinBERT or similar) into the dataset.
- Provides helper functions:
  - `add_text_data(df)`: Adds probability and data type columns to the dataframe by
    reading year-specific probability CSVs.
  - `getProbabilities(date, gvkey, df)`: Retrieves probability tuples for a given
    date and company identifier.

Dependencies:
    - pandas (optionally accelerated with cuDF)
    - numpy
    - torch
    - ast
    - math
"""
import ast

import cudf
import numpy as np


try:
    import cudf.pandas
    cudf.pandas.install()
    print("cuDF pandas accelerator mode enabled.")
except ImportError:
    print("cuDF not available, falling back to CPU pandas.")

import pandas as pd  # This import is now GPU-accelerated

# data = r"/teamspace/studios/this_studio/ret_sample.csv"
# print("starting to read csv")
# cols_to_drop = {} #columns we don't want
# usecols = lambda c: c not in cols_to_drop
# df = pd.read_csv(data, low_memory=False, parse_dates=["ret_eom", "date", "char_eom", "char_date"])
# print("done reading csv")
# get only rows who are USA companies
# df_trim = df[df["excntry"].astype(str).str.strip().str.upper().eq("USA")]
# print(df_trim.head())
# input("press key to continue")
# filename = r"/teamspace/studios/this_studio/retsamplepickle.pkl"
# df_trim.to_pickle(filename)

# def addTextData(df):
#
#   df["P1_"] = np.nan
#   df["P2_"] = np.nan
#   df["P3_"] = np.nan
#   df["dataType"] = np.nan
#
#   if 'date' in df.columns and 'gvkey' in df.columns:
#       for i in range(df.shape[0]):
#           date  = df.iat[i, df.columns.get_loc("date")]
#           gvkey = df.iat[i, df.columns.get_loc("gvkey")]
#
#           p1, p2, p3, dataType = getProbabilities(date, gvkey)
#
#           if i%1000 == 0: print(i)
#           # Assign values into new columns
#           df.at[i, "P1_"] = p1
#           df.at[i, "P2_"] = p2
#           df.at[i, "P3_"] = p3
#           df.at[i, "dataType"] = dataType
#
#   return df


def add_text_data(df):
    """
    Adds text-based probability features (P1_, P2_, P3_) and a data type flag to a DataFrame.

    This function looks up probability values for each row based on its `date` and `gvkey`
    from precomputed year-specific CSV files (e.g., '2005Probs.csv') located in the
    PKLcsvs directory. Each probability corresponds to a sentiment or classification score
    for that company and date.

    Args:
        df (pd.DataFrame): The input DataFrame containing at least 'date' and 'gvkey' columns.

    Returns:
        pd.DataFrame: The modified DataFrame with added columns:
            - P1_, P2_, P3_: float columns with probability values.
            - dataType: float column (1.0 for 10-K data, 0.0 otherwise).
    """
    if "date" not in df.columns or "gvkey" not in df.columns:
        return df

    # make sure row indexing is positional 0..n-1
    df = df.reset_index(drop=True)

    # pre-create with stable dtypes (avoid object)
    df["P1_"] = np.nan
    df["P2_"] = np.nan
    df["P3_"] = np.nan
    # keep dataType as float (0.0/1.0) so NaN is representable without pandas
    # Int64
    df["dataType"] = np.nan

    current_year = 2005
    path = f"/teamspace/studios/this_studio/PKLcsvs/{2005}Probs.csv"
    dataFrame = pd.read_csv(path, dtype=str).set_index("Date|gvkeys: ")
    p1s, p2s, p3s, flags = [], [], [], []
    dates = pd.to_datetime(df["char_eom"])
    years = dates.dt.year.to_numpy()
    months = dates.dt.month.to_numpy()
    gvkeys = df["gvkey"].astype(str).tolist()
    for i in range(df.shape[0]):

        gvkey = gvkeys[i]
        data_year = years[i]
        data_month = months[i]
        if data_year != current_year:
            path = f"/teamspace/studios/this_studio/PKLcsvs/{data_year}Probs.csv"
            dataFrame = pd.read_csv(path, dtype=str).set_index("Date|gvkeys: ")
            current_year = data_year
        # open currect file and initialize dataframe
        p1, p2, p3, flag = getProbabilities(data_year,
                                            data_month,
                                            gvkey,
                                            dataFrame)  # should be numbers

        if i % 1000 == 0:
            print(i)
        if i % 100000 == 0 and i != 0:
            # save data in new pkl
            k = len(p1s)  # rows filled so far
            idx = df.index[:k]

            block = np.column_stack((
                np.asarray(p1s, dtype="float64"),
                np.asarray(p2s, dtype="float64"),
                np.asarray(p3s, dtype="float64"),
                np.asarray(flags, dtype="float64"),
            ))

            df.loc[idx, ["P1_", "P2_", "P3_", "dataType"]] = block
            filename = r"/teamspace/studios/this_studio/IntermediateTrainingDataPickle.pkl"
            df.to_pickle(filename)
            print("Intermideately wrote to pickle. i = " + str(i))

        p1s.append(float(p1))
        p2s.append(float(p2))
        p3s.append(float(p3))
        flags.append(float(flag))
        # write as floats to keep column dtype numeric
    # lock dtypes explicitly (helps cuDF)
    df["P1_"] = np.asarray(p1s, dtype="float64")
    df["P2_"] = np.asarray(p2s, dtype="float64")
    df["P3_"] = np.asarray(p3s, dtype="float64")
    df["dataType"] = np.asarray(flags, dtype="float64")
    return df

# call with format: year, month, gvkey all integers with a
# .0 at end of gvkey. df is the dataframe its reading from


def getProbabilities(year, month, gvkey, df):
    """
    Retrieves a probability tuple (P1, P2, P3, dataType) for a given date and company ID.

    The function searches within the provided DataFrame (read from a year-specific
    probability CSV) for the matching year-month and gvkey. If found, it parses the
    stored stringified list of probabilities and returns numeric values. If missing or
    invalid, it returns a default uniform distribution and a dataType of 0.

    Args:
        date (str or pd.Timestamp): The target date for lookup.
        gvkey (float or str): The company identifier.
        df (pd.DataFrame): DataFrame indexed by 'YYYYMM' keys with gvkeys as columns.

    Returns:
        tuple[float, float, float, int]:
            (P1, P2, P3, dataType), where dataType = 1 if source is '10K', else 0.
    """
    DEFAULT = (0, 0, 0, 0)

    # convert date to a timestamp
    # if not isinstance(date, pd.Timestamp):
    #     try:
    #         date = pd.to_datetime(date)
    #     except Exception:
    #         print("date is not of type datetime (and couldn't be parsed).")
    #         return DEFAULT
    # year = date.year
    # ym_key = f"{year}{date.month:02d}"
    ym_key = f"{year}{month:02d}"

    if ym_key not in df.index or str(gvkey) not in df.columns:

        return DEFAULT
    cell = str(df.at[ym_key, str(gvkey)])

    # if cell is any of the below:
    # None or
    # (isinstance(cell, float) and math.isnan(cell)) or
    # (isinstance(cell, str) and cell.strip() == "")
    j = 1
    while (cell == 'nan' or cell == 'None') and (j <= month - 1):
        cell = str(df.at[f"{year}{month - j:02d}", str(gvkey)])
        j += 1
    if cell == 'None' or cell == 'nan':
        return DEFAULT
    data = ast.literal_eval(cell)
    data[3] = 1 if data[3] == '10K' else 0
    return data

# expanding window training loop
# for i in range (2012,2013):
#     df_train, df_val = xg_boost.initializeDataFrames(df, 2005, 2013)
#     r1_dates, r2_dates = xg_boost.get_regime_dates("2005-01-01", "2012-12-31")
#     model_r1, model_r2 = xg_boost.trainExperts(df_train, r1_dates, r2_dates)
#     need to use GPU to run above ^
#     input("Experts trained on 2005 - " + str(i) + "...")


if __name__ == "__main__":
    # #load ret sample data from pickle
    # data_frame_file = r"/teamspace/studios/this_studio/retsamplepickle.pkl"
    # df = pd.read_pickle(data_frame_file)
    # if not is_datetime(df["ret_eom"]):
    #     # If values look like '2005-01-31', this is enough:
    #     df["ret_eom"] = pd.to_datetime(df["ret_eom"], errors="coerce")
    # #add the finbert stuff and write to new pickle
    # filename = r"/teamspace/studios/this_studio/TrainingDataPickle.pkl"
    # trainingData = add_text_data(df)
    # trainingData.to_pickle(filename)

    # filename = r"/teamspace/studios/this_studio/PKLcsvs/2005Probs.csv"
    # df = pd.read_csv(filename, dtype=str).set_index("Date|gvkeys: ")
    # print(getProbabilities("2005-02-01", 1082.0, df))
    filename = r"/teamspace/studios/this_studio/TrainingDataPickle.pkl"
    df = pd.read_pickle(filename)
    print(df.head(n=1000000))
