"""
Module: build_probs_for

This module constructs regime probability tables for a given set of firm identifiers (gvkeys)
and dates by aggregating data from yearly probability CSV files. Each file is expected to 
contain serialized probability tuples for multiple firms, typically of the form:

    [P1, P2, P3, tag]

where `P1`, `P2`, and `P3` represent regime probabilities, and `tag` identifies the document 
type (e.g., "10K").

The module provides two main functions:
- `parse_cell(cell)`: Safely parses and validates a probability tuple, applying defaults
  when values are missing or malformed.
- `build_probs_for(df, base)`: Loads all relevant yearly CSVs, extracts the firm and date 
  data needed based on an input DataFrame, parses all probability cells, and returns a 
  consolidated long-format DataFrame with numeric probabilities and a binary tag indicator.

Typical usage example
---------------------
>>> import pandas as pd
>>> from build_probs_for import build_probs_for
>>> df = pd.DataFrame({"date": ["2007-05-01"], "gvkey": ["12345"]})
>>> probs = build_probs_for(df, base="/path/to/PKLcsvs")
>>> print(probs.head())

Output columns:
    - ym:        Year-month key (string)
    - gvkey:     Firm identifier
    - P1_, P2_, P3_: Regime probabilities
    - dataType:  Binary flag (1.0 if tag == "10K", else 0.0)

Notes
-----
- Invalid or missing cells default to (1/3, 1/3, 1/3, 0.0).
- Only `gvkey`s present in the input DataFrame are retained.
- CSV files are assumed to follow the naming convention "YYYYProbs.csv".
"""
import ast
from pathlib import Path
import pandas as pd

DEFAULT = (1/3, 1/3, 1/3, 0.0)

def parse_cell(cell):
    """
    Safely parse and validate a probability cell from a CSV file.

    This function handles various possible representations of a probability tuple:
    - A stringified list or tuple (e.g., "[0.2, 0.5, 0.3, '10K']")
    - A native Python list or tuple
    - Missing values (NaN)

    It ensures that the output is always a 4-element tuple of the form:

        (P1, P2, P3, dataType)

    where P1, P2, and P3 are floats representing probabilities, and dataType is a binary 
    flag indicating whether the fourth element of the cell corresponds to the tag "10K".

    Parameters
    ----------
    cell : object
        A single cell value from the probability CSV. May be a string, list, tuple,
        or NaN.

    Returns
    -------
    tuple of float
        A 4-tuple `(P1, P2, P3, flag)` where:
        - `P1`, `P2`, `P3` are floats (defaulting to 1/3 each if invalid).
        - `flag` is 1.0 if the tag equals "10K" (case-insensitive), otherwise 0.0.

    Notes
    -----
    - Invalid, malformed, or missing cells return the default `(1/3, 1/3, 1/3, 0.0)`.
    - The function uses `ast.literal_eval` for safe parsing of stringified lists.
    """
    # Accept NaN, string like "[...]", or actual list/tuple
    if pd.isna(cell):
        return DEFAULT
    if isinstance(cell, str):
        try:
            cell = ast.literal_eval(cell)
        except Exception:
            return DEFAULT
    if not isinstance(cell, (list, tuple)) or len(cell) != 4:
        return DEFAULT
    try:
        p1, p2, p3 = float(cell[0]), float(cell[1]), float(cell[2])
    except Exception:
        return DEFAULT
    tag = cell[3]
    flag = 1.0 if (isinstance(tag, str) and tag.strip().upper() == "10K") else 0.0
    return (p1, p2, p3, flag)

def build_probs_for(df, base="/teamspace/studios/this_studio/PKLcsvs"):
    """
    Build a consolidated probability DataFrame from yearly CSV files.

    This function reads regime probability CSV files (e.g., "2005Probs.csv") from a 
    specified directory and extracts only the rows and columns corresponding to the 
    `date` and `gvkey` values present in the given DataFrame `df`. Each cell in the 
    source files is expected to contain a serialized 4-tuple of the form:
        [P1, P2, P3, tag]
    where `tag` typically denotes a document type such as "10K".

    The function parses these cells, replaces invalid or missing entries with a default
    value `(1/3, 1/3, 1/3, 0.0)`, and returns a long-format DataFrame with one row per 
    `(ym, gvkey)` pair, containing numeric columns for each probability component and 
    a binary flag for document type.

    Parameters
    ----------
    df : pandas.DataFrame
        Input DataFrame containing at least two columns:
        - 'date': used to determine which years' CSVs to load.
        - 'gvkey': used to select relevant firm identifiers.
    base : str or pathlib.Path, optional
        Directory path containing the yearly probability CSVs.
        Defaults to "/teamspace/studios/this_studio/PKLcsvs".

    Returns
    -------
    pandas.DataFrame
        A long-format DataFrame with columns:
        - 'ym': year-month key (string)
        - 'gvkey': firm identifier (string)
        - 'P1_', 'P2_', 'P3_': regime probabilities (float64)
        - 'dataType': binary flag (1.0 if tag == "10K", else 0.0)

    Notes
    -----
    - Missing, malformed, or unparsable cells are replaced with the default tuple.
    - Only the `gvkey`s found in the input DataFrame are retained to reduce memory usage.
    - This function assumes CSV files are named like "YYYYProbs.csv" and contain
      one column per gvkey.
    """
    years   = sorted(pd.to_datetime(df["date"]).dt.year.unique())
    need_gv = set(df["gvkey"].astype(str).unique())

    frames = []
    for y in years:
        path = Path(base) / f"{y}Probs.csv"
        t = pd.read_csv(path, dtype=object)

        # Rename the first column to 'ym'
        first = t.columns[0]
        if first != "ym":
            t = t.rename(columns={first: "ym"})
        t["ym"] = t["ym"].astype(str)

        # Keep only the gvkeys we need (cuts memory & time)
        keep = ["ym"] + [c for c in t.columns[1:] if str(c) in need_gv]
        t = t[keep]

        # Wide -> long: (ym, gvkey, cell)
        long = t.melt(id_vars="ym", var_name="gvkey", value_name="cell")
        long["gvkey"] = long["gvkey"].astype(str)

        # Parse the 4-tuple once
        parsed = long["cell"].map(parse_cell)
        long[["P1_", "P2_", "P3_", "dataType"]] = pd.DataFrame(parsed.tolist(), index=long.index)
        frames.append(long.drop(columns=["cell"]))

    if not frames:
        return pd.DataFrame(columns=["ym", "gvkey", "P1_", "P2_", "P3_", "dataType"])

    probs = pd.concat(frames, ignore_index=True)
    probs[["P1_", "P2_", "P3_", "dataType"]] = (
        probs[["P1_", "P2_", "P3_", "dataType"]].astype("float64")
    )
    return probs
