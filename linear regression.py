import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.ticker import FormatStrFormatter
import math
# Load Excel file
file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Dataset.xlsx"
df = pd.read_excel(file_path)

# Define predictors and targets
predictors = [ "fiber_length","fiber_width","fiber_factor", "fiber_branch","fiber_area"]
targets = ["A.I. Fracture Stress","A.I. Fracture Strain", "A.I. Failure Stress",
           "A.I. Failure Strain", "A.I. Fracture Length", "A.I. Young's Modulus","A.I. Toughness"]

# Ensure predictors and targets exist in the dataset
missing_predictors = [col for col in predictors if col not in df.columns]
missing_targets = [col for col in targets if col not in df.columns]

if missing_predictors or missing_targets:
    print(f"Error: Missing Columns: {missing_predictors + missing_targets}")
    exit()

# Drop NaN values (if any)
df = df.dropna()

# Select valid data
X = df[predictors]
y = df[targets]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on train and test sets
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# Calculate Training Errors
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

# Calculate Testing Errors
test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print("\n  Training Errors:")
print(f"  - Mean Absolute Error (MAE): {train_mae:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {train_rmse:.4f}")
print(f"  - R-squared (R²): {train_r2:.4f}")

print("\n  Testing Errors:")
print(f"  - Mean Absolute Error (MAE): {test_mae:.4f}")
print(f"  - Root Mean Squared Error (RMSE): {test_rmse:.4f}")
print(f"  - R-squared (R²): {test_r2:.4f}")

# Prepare data for Excel output
excel_data = {}
for i, target in enumerate(targets):
    excel_data[f"{target}_Observed"] = y_test[target].values
    excel_data[f"{target}_Predicted"] = y_test_pred[:, i]
    excel_data[f"{target}_Train_R2"] = [r2_score(y_train[target], y_train_pred[:, i])]*len(y_test[target])
    excel_data[f"{target}_Test_R2"] = [r2_score(y_test[target], y_test_pred[:, i])]*len(y_test[target])

# Create a DataFrame for output
output_df = pd.DataFrame(excel_data)

# Save the results to Excel
output_file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Linear_Regression_Results.xlsx"
output_df.to_excel(output_file_path, index=False)

print(f"\n Data saved to: {output_file_path}")

# Determine the number of target variables
num_targets = len(targets)
cols = 3  # Define the number of columns for the subplot grid
rows = math.ceil(num_targets / cols)  # Calculate the required number of rows

# Set the figure size dynamically based on the grid dimensions
plt.figure(figsize=(cols * 6, rows * 6))

for i, target in enumerate(targets):
    ax = plt.subplot(rows, cols, i + 1)

    # Calculate the R² score for the current target
    r2_target_test = r2_score(y_test[target], y_test_pred[:, i])

    # Create a scatter plot with observed vs. predicted values
    sns.scatterplot(x=y_test[target], y=y_test_pred[:, i], s=50, color="blue", label="Predicted", ax=ax)

    # Plot the y = x reference line for perfect predictions
    min_val = min(y_test[target].min(), y_test_pred[:, i].min())
    max_val = max(y_test[target].max(), y_test_pred[:, i].max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', label="Perfect Fit")

    # Set axis labels and title with R² score
    ax.set_xlabel("Observed Values", fontsize=14)
    ax.set_ylabel("Predicted Values", fontsize=14)
    ax.set_title(f"{target}\nR²: {r2_target_test:.4f}", fontsize=16)

    # Add legend and grid for better readability
    ax.legend()
    ax.grid(True)

    # Format axis tick labels to display two decimal places
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    # Ensure equal aspect ratio for accurate data representation
    ax.set_aspect('equal', adjustable='datalim')

# Adjust layout to prevent overlap and improve spacing
plt.subplots_adjust(hspace=0.5, wspace=0.4)

# Save the figure as a high-quality image file
plt.savefig(r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Observed_vs_Predicted.png", dpi=300, bbox_inches="tight")

# Display the plots
plt.show()