print("Training Random Forest for feature importance...")

# Keep RF moderate so it runs in Colab
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    class_weight="balanced_subsample"
)

rf.fit(X_train_scaled, y_train)

rf_scores = rf.feature_importances_

rf_df = pd.DataFrame({
    "feature": feature_cols,
    "rf_score": rf_scores
}).sort_values("rf_score", ascending=False).reset_index(drop=True)

rf_df["rf_rank"] = np.arange(1, len(rf_df) + 1)
