# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
# import kagglehub
# from kagglehub import KaggleDatasetAdapter

# # Set the path to the file you'd like to load
# file_path = ""

# # Load the latest version
# df = kagglehub.load_dataset(
#   KaggleDatasetAdapter.PANDAS,
#   "kmader/food41",
#   file_path,
#   # Provide any additional arguments like 
#   # sql_query or pandas_kwargs. See the 
#   # documenation for more information:
#   # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
# )

# print("First 5 records:", df.head())
# -----------------------------------
# import kagglehub

# # Download latest version
# path = kagglehub.dataset_download("kmader/food41")

# print("Path to dataset files:", path)

import os

dataset_path = "/root/.cache/kagglehub/datasets/kmader/food41/versions/5"

# List files/folders
for root, dirs, files in os.walk(dataset_path):
    print(f"Current Directory: {root}")
    print(f"Subdirs: {dirs}")
    print(f"Files: {files}")
    break  # Remove this if you want to explore deeper
