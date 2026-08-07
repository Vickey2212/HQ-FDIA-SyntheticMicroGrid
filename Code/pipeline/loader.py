train_labels_small = [train_full.df.iloc[i][LABEL_COL] for i in train_small.indices]
class_counts = Counter(train_labels_small)
print("Small-train class distribution:", class_counts)

class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
sample_weights = [class_weights[label] for label in train_labels_small]
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_small,
    batch_size=16,   # keep small for quantum
    sampler=sampler,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    val_small,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_batch
)

test_loader = DataLoader(
    test_small,
    batch_size=16,
    shuffle=False,
    collate_fn=collate_batch
)
