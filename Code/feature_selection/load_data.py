df = pd.read_csv(CSV_PATH, low_memory=False)

assert LABEL_COL in df.columns, f"Missing label column: {LABEL_COL}"

print("Original shape:", df.shape)
print("Label distribution:")
print(df[LABEL_COL].value_counts())
