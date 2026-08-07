selected_features = rank_df["feature"].head(TOP_K).tolist()

print(f"\nSelected top {TOP_K} features:")
for i, f in enumerate(selected_features[:20], start=1):
    print(f"{i:02d}. {f}")
if len(selected_features) > 20:
    print("...")
