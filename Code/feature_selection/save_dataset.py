train_selected = pd.concat(
    [X_train_scaled[selected_features].reset_index(drop=True),
     y_train.reset_index(drop=True)],
    axis=1
)

val_selected = pd.concat(
    [X_val_scaled[selected_features].reset_index(drop=True),
     y_val.reset_index(drop=True)],
    axis=1
)

full_selected = pd.concat(
    [X_full_scaled[selected_features].reset_index(drop=True),
     y.reset_index(drop=True)],
    axis=1
)

train_selected.to_csv(TRAIN_OUT, index=False)
val_selected.to_csv(VAL_OUT, index=False)
full_selected.to_csv(FULL_OUT, index=False)

rank_df.to_csv(RANKING_CSV, index=False)

with open(FEATURES_JSON, "w") as f:
    json.dump({
        "label_col": LABEL_COL,
        "top_k": TOP_K,
        "selected_features": selected_features
    }, f, indent=2)
