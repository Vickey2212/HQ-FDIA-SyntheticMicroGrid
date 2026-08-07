print("Computing Mutual Information...")
mi_scores = mutual_info_classif(
    X_train_scaled,
    y_train,
    discrete_features=False,
    random_state=RANDOM_STATE
)

mi_df = pd.DataFrame({
    "feature": feature_cols,
    "mi_score": mi_scores
}).sort_values("mi_score", ascending=False).reset_index(drop=True)

mi_df["mi_rank"] = np.arange(1, len(mi_df) + 1)
