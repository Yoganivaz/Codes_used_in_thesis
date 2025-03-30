import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Load Excel file
file_path = "E:\Desktop\Thesis\Actual tensile strength $ -Final\Variance.xlsx"  # Update with your file path
sheet_name = "Sheet1"  # Update with the correct sheet name if necessary

# Read data
df = pd.read_excel(file_path, sheet_name=sheet_name)

# Normalize data (Min-Max Scaling)
scaler = MinMaxScaler()
normalized_data = scaler.fit_transform(df)

# Convert back to DataFrame
df_normalized = pd.DataFrame(normalized_data, columns=df.columns)

# Calculate variance for each parameter (column)
variance_values = df_normalized.var()

# Add variance values to the Excel sheet
df_variance = pd.DataFrame(variance_values, columns=["Variance"])

# Write back to the same Excel file
with pd.ExcelWriter(file_path, engine="openpyxl", mode="a") as writer:
    df_variance.to_excel(writer, sheet_name="Variance_Results")

print("Variance calculation completed. Check the Excel file for results.")
