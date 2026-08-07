# transpose so duplicated() checks columns by content
dup_mask = work_df[feature_cols].T.duplicated()
duplicate_cols = work_df[feature_cols].T[dup_mask].index.tolist()

print("Removed duplicate features:", len(duplicate_cols))

feature_cols = [c for c in feature_cols if c not in duplicate_cols]
work_df = work_df[feature_cols + [LABEL_COL]].copy()

print("Feature count after cleanup:", len(feature_cols))
