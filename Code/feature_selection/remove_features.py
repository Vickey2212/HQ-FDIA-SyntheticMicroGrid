nunique = work_df[measurement_cols].nunique()
non_constant_cols = nunique[nunique > 1].index.tolist()

removed_constant = sorted(set(measurement_cols) - set(non_constant_cols))
print("Removed constant features:", len(removed_constant))

work_df = work_df[non_constant_cols + [LABEL_COL]].copy()
feature_cols = non_constant_cols
