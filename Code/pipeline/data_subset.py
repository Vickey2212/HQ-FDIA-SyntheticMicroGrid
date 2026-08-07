def make_stratified_subset(dataset, label_col, n_samples, seed=7):
    labels = dataset.df[label_col].tolist()
    indices = list(range(len(labels)))

    idx_small, _ = train_test_split(
        indices,
        train_size=n_samples,
        stratify=labels,
        random_state=seed
    )
    return Subset(dataset, idx_small)

# ---------------------------------
# load full datasets
# ---------------------------------
train_full = SelectedGraphFDIADataset(TRAIN_CSV_PATH, EDGE_LIST_1BASED, N_BUSES, LABEL_COL)
val_full   = SelectedGraphFDIADataset(VAL_CSV_PATH, EDGE_LIST_1BASED, N_BUSES, LABEL_COL)

# ---------------------------------
# small train subset = 2000
# ---------------------------------
train_small = make_stratified_subset(train_full, LABEL_COL, n_samples=1600, seed=7)

# ---------------------------------
# split validation file into 200 val + 200 test
# ---------------------------------
val_labels = val_full.df[LABEL_COL].tolist()
val_indices = list(range(len(val_labels)))

val_idx_400, _ = train_test_split(
    val_indices,
    train_size=400,
    stratify=val_labels,
    random_state=7
)

temp_labels = [val_labels[i] for i in val_idx_400]

val_idx_200, test_idx_200 = train_test_split(
    val_idx_400,
    train_size=200,
    stratify=temp_labels,
    random_state=7
)

val_small = Subset(val_full, val_idx_200)
test_small = Subset(val_full, test_idx_200)
