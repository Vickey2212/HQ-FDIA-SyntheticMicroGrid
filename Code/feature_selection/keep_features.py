measurement_cols = [
    c for c in df.columns
    if c.startswith("Flow_") or c.startswith("Inj_")
]

print("Measurement feature count:", len(measurement_cols))

# Convert measurement cols to numeric safely
df[measurement_cols] = df[measurement_cols].apply(pd.to_numeric, errors="coerce")

# Replace inf with NaN, then fill NaN
df[measurement_cols] = df[measurement_cols].replace([np.inf, -np.inf], np.nan)
df[measurement_cols] = df[measurement_cols].fillna(0.0)

# Keep only features + label
work_df = df[measurement_cols + [LABEL_COL]].copy()

print("Working shape:", work_df.shape)
