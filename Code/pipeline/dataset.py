class SelectedGraphFDIADataset(Dataset):
    def __init__(self, csv_path, edge_list_1based, n_buses, label_col):
        self.df = pd.read_csv(csv_path, low_memory=False)
        assert label_col in self.df.columns, f"Missing label column: {label_col}"

        self.edge_list_1based = edge_list_1based
        self.n_buses = n_buses
        self.label_col = label_col

        measurement_cols = [
            c for c in self.df.columns
            if c.startswith("Flow_") or c.startswith("Inj_")
        ]

        self.df[measurement_cols] = self.df[measurement_cols].apply(
            pd.to_numeric, errors="coerce"
        )
        self.df[measurement_cols] = self.df[measurement_cols].replace([np.inf, -np.inf], np.nan)
        self.df[measurement_cols] = self.df[measurement_cols].fillna(0.0)

        self.flows_map, self.inj_map, self.T = build_index_maps(self.df.columns)

        print(f"Loaded {csv_path}")
        print("Rows:", len(self.df))
        print("Selected measurement cols:", len(measurement_cols))
        print("Detected time window T:", self.T)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        node_ts, edge_ts = extract_time_series_row(
            row, self.flows_map, self.inj_map, self.T,
            self.edge_list_1based, self.n_buses
        )
        y = int(row[self.label_col])

        return (
            torch.tensor(node_ts, dtype=torch.float32),
            torch.tensor(edge_ts, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long)
        )

def collate_batch(batch):
    node_ts = torch.stack([b[0] for b in batch], dim=0)
    edge_ts = torch.stack([b[1] for b in batch], dim=0)
    y = torch.stack([b[2] for b in batch], dim=0)
    return node_ts, edge_ts, y
