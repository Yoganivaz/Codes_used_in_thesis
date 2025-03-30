import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import linregress
from scipy.integrate import simpson
import os


def analyze_stress_strain(file_path):
    if not os.path.exists(file_path):
        print("File not found! Please check the directory and file name.")
        return

    try:
        data = pd.read_excel(file_path)
        print("Columns in the Excel file:", data.columns.tolist())

        strain_col = 'True Strain'  # Replace with actual column name
        stress_col = 'True Stress'  # Replace with actual column name

        if strain_col not in data.columns or stress_col not in data.columns:
            print(f"Error: '{strain_col}' and/or '{stress_col}' columns not found in the Excel file.")
            return

        strain = data[strain_col].values
        stress = data[stress_col].values

        if len(strain) == 0 or len(stress) == 0:
            print("Error: Strain and/or Stress data is empty.")
            return

        fracture_stress = np.max(stress)
        fracture_index = np.argmax(stress)
        fracture_strain = strain[fracture_index]

        failure_stress = stress[-1]
        failure_strain = strain[-1]

        def find_linear_region(strain, stress, max_points=50, tolerance=0.02):
            for end in range(max_points, len(strain)):
                subset_strain = strain[:end]
                subset_stress = stress[:end]
                slope, intercept, r_value, _, _ = linregress(subset_strain, subset_stress)
                if abs(r_value) < (1 - tolerance):
                    return end - 1
            return max_points

        linear_end = find_linear_region(strain, stress)
        strain_elastic = strain[:linear_end]
        stress_elastic = stress[:linear_end]

        slope, intercept, _, _, _ = linregress(strain_elastic, stress_elastic)
        youngs_modulus = slope

        fracture_length = failure_strain - fracture_strain
        toughness = simpson(y=stress, x=strain)

        print("\nComputed Parameters:")
        print(f"Fracture Stress (Pa): {fracture_stress}")
        print(f"Fracture Strain: {fracture_strain}")
        print(f"Failure Stress (Pa): {failure_stress}")
        print(f"Failure Strain: {failure_strain}")
        print(f"Young's Modulus (Pa): {youngs_modulus}")
        print(f"Fracture Length: {fracture_length}")
        print(f"Toughness (Pa): {toughness}")

        plt.figure(figsize=(10, 6))
        plt.plot(strain, stress, label="Stress-Strain Curve", color="blue")
        plt.fill_between(strain, stress, color='skyblue', alpha=0.4, label="Toughness Area")
        plt.scatter([fracture_strain], [fracture_stress], color="red", label="Fracture Point")
        plt.scatter([failure_strain], [failure_stress], color="green", label="Failure Point")
        plt.title("True Stress vs True Strain")
        plt.xlabel("True Strain")
        plt.ylabel("True Stress (kPa)")
        plt.legend()
        plt.grid()

        plot_file_name = os.path.splitext(file_path)[0] + "_stress_strain_curve.png"
        plt.savefig(plot_file_name)
        plt.show()

        results = {
            "Parameter": [
                "Fracture Stress (kPa)",
                "Fracture Strain",
                "Failure Stress (kPa)",
                "Failure Strain",
                "Young's Modulus (kPa)",
                "Fracture Length",
                "Toughness"
            ],
            "Value": [
                fracture_stress,
                fracture_strain,
                failure_stress,
                failure_strain,
                youngs_modulus,
                fracture_length,
                toughness
            ]
        }

        results_df = pd.DataFrame(results)
        with pd.ExcelWriter(file_path, engine="openpyxl", mode="a", if_sheet_exists="replace") as writer:
            results_df.to_excel(writer, sheet_name="Results", index=False)

        print(f"\nAnalysis complete! Results saved to '{file_path}' in the 'Results' sheet.")
        print(f"Stress-strain curve plot saved as '{plot_file_name}'.")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    file_path = input("Enter the full path to your Excel file: ")
    analyze_stress_strain(file_path)