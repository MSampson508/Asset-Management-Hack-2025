"""
A script for performing batch financial sentiment analysis using FinBERT.

This module loads a pre-trained FinBERT model and tokenizer, utilizes a GPU 
(if available) for acceleration, and processes management discussion (mgmt) 
text from a pickled pandas DataFrame. The script calculates the FinBERT 
probability triplet (positive, negative, neutral) for each text entry 
and outputs the results into a wide-format CSV file.

The output CSV is structured with dates as rows, gvkeys (company identifiers) 
as columns, and the cell data containing the sentiment probabilities plus 
the 'file_type' for the corresponding gvkey/date pair.

"""

import pickle
import csv
from collections import defaultdict
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

#ADDED FOR GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


# Load the FinBERT model and tokenizer
tokenizer = AutoTokenizer.from_pretrained('ProsusAI/finbert')
model = AutoModelForSequenceClassification.from_pretrained('ProsusAI/finbert')

#ADDED FOR GPU
model.to(device)


def get_probabilities(text):
    """
    Compute model output probabilities for a given input text.

    This function tokenizes the input string using a pre-defined tokenizer,
    runs it through a preloaded model, and returns the softmax-normalized
    probabilities of the model’s output logits.

    Parameters
    ----------
    text : str
        The input text to be evaluated. Must be a valid string.

    Returns
    -------
    list of float
        A list of probabilities corresponding to the model’s output classes.

    Notes
    -----
    - If the input is not a string, the function returns None.
    - The tokenizer and model must be defined in the global scope.
    - The model and input tensors are assumed to be on the same device.
    """
    if isinstance(text, str):
        inputs = tokenizer(
            text,
            add_special_tokens=True,
            padding=True,
            truncation=True,
            return_tensors='pt'
        ).to(device)
        outputs = model(**inputs)
        probabilities = torch.nn.functional.softmax(outputs.logits, dim=-1)
        # Have to splice the probabilities to get only numbers
        return probabilities.squeeze().tolist()


#want to have a csv output with the format:
#                   gvkey1  gvkey2 ...
#date1 20050228     x1      x2      ...
#date2 20050330     x1      x2      ...
#where xi is a finbert triplet of probabilities
FILE_PATH_OUT = r"/teamspace/studios/this_studio/PKLcsvs/2024Probs.csv"
FILE_PATH_IN = r"/teamspace/studios/this_studio/ZippedPKLs/Raw_PKLs/text_us_2024.pkl"
table = defaultdict(dict)
gvkeys = set()
dates = set()

#read the pkl file
with open(FILE_PATH_IN, "rb") as f:   # open the file in binary read mode
    data = pickle.load(f)

#read data
COUNT = 0
for index, row in data.iterrows():
    if COUNT%1000 == 0:
        print(COUNT)
    COUNT += 1

    if (
        ('date' in data.columns) and
        ('gvkey' in data.columns) and
        ('mgmt' in data.columns) and
        ('file_type' in data.columns)
    ):
        mgmt = row["mgmt"]
        gvkey = row["gvkey"]
        DATE = row["date"]

        DATESTR = str(DATE)
        DATESTR = DATESTR[:-2]
        DATE = int(DATESTR)

        type_append = row["file_type"]
        tableData = get_probabilities(mgmt)
        tableData.append(type_append)
        table[DATE][gvkey] =  tableData
        dates.add(DATE)
        gvkeys.add(gvkey)
gvkeys = sorted(gvkeys)
dates = sorted(dates)

#write to csv in correct place
with open(FILE_PATH_OUT, mode="w", newline="", encoding="utf-8") as file:
    writer = csv.writer(file)
    writer.writerow(["Date|gvkeys: "] + gvkeys)
    for DATE in dates:
        row_out = [DATE]
        for comp in gvkeys:
            row_out.append(table[DATE].get(comp, ""))  # empty if no data
        writer.writerow(row_out)
