X = work_df[feature_cols].copy()
y = work_df[LABEL_COL].copy()

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

print("Train shape:", X_train.shape)
print("Val shape:", X_val.shape)
