"""
Module to inspect the contents of a pickled dataset file.

This utility script safely loads a pickle file and reports its structure
and contents. It is primarily intended for debugging or verifying the
format of serialized data (such as `retsamplepickle.pkl`) before using it
in a machine learning or data processing pipeline.

The script:
- Opens the pickle file located at `RET_PKL`.
- Determines whether the object is a pandas DataFrame, dictionary, or another type.
- Prints relevant summary information such as column names, dictionary keys,
  or the first few items of the data.

Usage
-----
Run this script directly to print a structured summary of the pickle file:

    $ python inspect_pickle.py

Notes
-----
- If the file contains a pandas DataFrame, the first few rows are printed.
- If the file contains a dictionary, only the top-level keys are shown.
- For other objects, a simple preview is attempted.
"""
import pickle
import pandas as pd

RET_PKL = "/teamspace/studios/this_studio/retsamplepickle.pkl"

with open(RET_PKL, "rb") as f:
    data = pickle.load(f)

print("Type:", type(data))

if isinstance(data, pd.DataFrame):
    print("Columns:", data.columns.tolist())
    print(data.head())
elif isinstance(data, dict):
    print("Keys:", list(data.keys()))
else:
    try:
        print("First 5 items:", data[:5])
    except Exception:
        print("Preview:", data)
