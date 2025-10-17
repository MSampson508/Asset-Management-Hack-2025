"""
A utility module for safely extracting the contents of a ZIP file.

This script defines the necessary paths for a target ZIP file and a destination 
directory, then performs the extraction operation using Python's built-in 
`zipfile` library.

"""
import zipfile

# Path to zip file (paste from "Copy path")
ZIP_PATH = r"/teamspace/studios/this_studio/ZippedPKLs/2025.zip"

# Folder to extract into
EXTRACT_DIR = r"/teamspace/studios/this_studio/ZippedPKLs/Raw_PKLs"

with zipfile.ZipFile(ZIP_PATH, 'r') as zip_ref:
    zip_ref.extractall(EXTRACT_DIR)
print("Extracted to:", EXTRACT_DIR)
