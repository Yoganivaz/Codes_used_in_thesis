import pandas as pd
import numpy as np
import os

# Load Excel file
file_path = "E:\Desktop\Thesis\Actual tensile strength $ -Final\Dataset.xlsx"
df = pd.read_excel(file_path)

# Shuffle data randomly (keeping index intact)
df_shuffled = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Define the row sizes for splitting
row_sizes = list(range(3, 34, 3))  # Generates [3, 6, 9, ..., 66]

# Save directory
save_path = "E:\\Desktop\\Thesis\\Actual tensile strength $ -Final\\data split\\"
os.makedirs(save_path, exist_ok=True)  # Ensure the directory exists

# Loop through row sizes and create subsets
for rows in row_sizes:
    subset = df_shuffled.iloc[:rows]  # Select the first 'rows' samples
    file_name = f"Dataset_{rows}_Rows.xlsx"
    subset.to_excel(os.path.join(save_path, file_name), index=False)

print(" All datasets saved successfully!")

