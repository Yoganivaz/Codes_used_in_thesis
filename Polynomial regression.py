import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from matplotlib.ticker import FormatStrFormatter
import math

# Load Excel file
file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Dataset.xlsx"   # Update with correct path
df = pd.read_excel(file_path)

# Define predictors and targets
predictors = ["SPI", "PPI", "WG"]
targets = ["A.I. Fracture Stress","A.I. Fracture Strain", "A.I. Failure Stress",
           "A.I. Failure Strain", "A.I. Fracture Length", "A.I. Young's Modulus","A.I. Toughness", "fiber_length","fiber_width","fiber_factor", "fiber_branch","fiber_area"]

# Ensure predictors and targets exist in the dataset
missing_predictors = [col for col in predictors if col not in df.columns]
missing_targets = [col for col in targets if col not in df.columns]

if missing_predictors:
    print(f"Error: Missing predictor columns: {missing_predictors}")
    exit()

if missing_targets:
    print(f"Error: Missing target columns: {missing_targets}")
    exit()

# Drop NaN values if any
df = df.dropna()

# Select valid data
X = df[predictors]
y = df[targets]

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Apply Polynomial Features (degree = 2)
degree = 4  # Change this for higher-degree polynomial regression
poly = PolynomialFeatures(degree=degree, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

# Initialize and train the polynomial regression model
model = LinearRegression()
model.fit(X_train_poly, y_train)

# Predict on train and test sets
y_train_pred = model.predict(X_train_poly)
y_test_pred = model.predict(X_test_poly)

# Calculate errors separately for train and test
train_mae = mean_absolute_error(y_train, y_train_pred)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

test_mae = mean_absolute_error(y_test, y_test_pred)
test_rmse = np.sqrt(mean_squared_error(y_test, y_test_pred))
test_r2 = r2_score(y_test, y_test_pred)

# Print results
print(f"\nPolynomial Regression (Degree {degree}) Evaluation:")
print(f"Train MAE: {train_mae:.4f}, Test MAE: {test_mae:.4f}")
print(f"Train RMSE: {train_rmse:.4f}, Test RMSE: {test_rmse:.4f}")
print(f"Train R²: {train_r2:.4f}, Test R²: {test_r2:.4f}")

# Save results to Excel (Fixing array length mismatch)
train_data = pd.DataFrame(y_train).reset_index(drop=True)
train_data["Train_Predicted"] = y_train_pred.tolist()

test_data = pd.DataFrame(y_test).reset_index(drop=True)
test_data["Test_Predicted"] = y_test_pred.tolist()

# Save the DataFrames to an Excel file
output_file_path = r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Polynomial_Regression_Results.xlsx"
with pd.ExcelWriter(output_file_path) as writer:
    train_data.to_excel(writer, sheet_name="Train Results", index=False)
    test_data.to_excel(writer, sheet_name="Test Results", index=False)

print(f"Data with predictions and errors saved to {output_file_path}")

# Determine the number of target variables
num_targets = len(targets)
cols = 3  # Define the number of columns for the subplot grid
rows = math.ceil(num_targets / cols)  # Calculate the required number of rows

#  Set the figure size dynamically based on the grid dimensions
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

    #  Ensure equal aspect ratio for accurate data representation
    ax.set_aspect('equal', adjustable='datalim')

#  Adjust layout to prevent overlap and improve spacing
plt.subplots_adjust(hspace=0.5, wspace=0.4)

#  Save the figure as a high-quality image file
plt.savefig(r"E:\Desktop\Thesis\Actual tensile strength $ -Final\Observed_vs_Predicted.png", dpi=300, bbox_inches="tight")

# Display the plots
plt.show()
