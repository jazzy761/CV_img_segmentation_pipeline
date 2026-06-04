import pandas as pd

df = pd.read_csv("Measurement\measurements.csv")

df["absolute_error"] = (
    df["ground_truth_mm"] -
    df["predicted_mm"]
).abs()

df["percentage_error"] = (
    df["absolute_error"] /
    df["ground_truth_mm"]
) * 100

mae = df["absolute_error"].mean()
mpe = df["percentage_error"].mean()

print("\nMeasurement Evaluation")
print("-" * 40)

print(df)

print("\nMAE (mm):", round(mae, 3))
print("MPE (%):", round(mpe, 3))


"""Note: The above values are illustrative examples demonstrating the evaluation methodology. 
Final MAE and MPE must be computed using physical caliper measurements collected during deployment validation."""