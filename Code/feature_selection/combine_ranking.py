rank_df = pd.merge(mi_df, rf_df, on="feature", how="inner")

# Normalize scores for easier comparison
rank_df["mi_score_norm"] = rank_df["mi_score"] / (rank_df["mi_score"].max() + 1e-12)
rank_df["rf_score_norm"] = rank_df["rf_score"] / (rank_df["rf_score"].max() + 1e-12)

# Combined score: average normalized importance
rank_df["combined_score"] = 0.5 * rank_df["mi_score_norm"] + 0.5 * rank_df["rf_score_norm"]

# Combined rank: smaller is better
rank_df["rank_sum"] = rank_df["mi_rank"] + rank_df["rf_rank"]

# Final sort
rank_df = rank_df.sort_values(
    ["combined_score", "rank_sum"],
    ascending=[False, True]
).reset_index(drop=True)

rank_df["final_rank"] = np.arange(1, len(rank_df) + 1)

print("\nTop 20 selected features preview:")
print(rank_df[["final_rank", "feature", "mi_score", "rf_score", "combined_score"]].head(20))
