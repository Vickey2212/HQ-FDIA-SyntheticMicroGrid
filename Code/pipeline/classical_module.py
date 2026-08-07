class SharedTCN(nn.Module):
    def __init__(self, out_channels=T_OUT, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv = nn.Conv1d(1, out_channels, kernel_size, padding=pad)
        self.bn = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, S, T)
        B, S, T = x.shape
        x = x.reshape(B * S, 1, T)
        h = F.relu(self.bn(self.conv(x)))   # (B*S, C, T)
        h = h.mean(dim=-1)                  # (B*S, C)
        return h.reshape(B, S, -1)          # (B, S, C)

class SimpleGNN(nn.Module):
    def __init__(self, n_buses, edge_list_1based, node_in_dim, edge_in_dim, hidden=NODE_HIDDEN, steps=GNN_STEPS):
        super().__init__()
        self.n_buses = n_buses
        self.edges = [(i - 1, j - 1) for (i, j) in edge_list_1based]
        self.steps = steps

        self.node_proj = nn.Linear(node_in_dim, hidden)
        self.edge_proj = nn.Linear(edge_in_dim, hidden)

        self.msg = nn.Sequential(
            nn.Linear(hidden * 2, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
        )
        self.gru = nn.GRUCell(hidden, hidden)

    def forward(self, node_x, edge_x):
        B, N, _ = node_x.shape
        node_h = self.node_proj(node_x)
        edge_h = self.edge_proj(edge_x)

        for _ in range(self.steps):
            agg = torch.zeros_like(node_h)
            for eidx, (u, v) in enumerate(self.edges):
                hu = node_h[:, u, :]
                hv = node_h[:, v, :]
                he = edge_h[:, eidx, :]

                m_uv = self.msg(torch.cat([hu, he], dim=-1))
                m_vu = self.msg(torch.cat([hv, he], dim=-1))

                agg[:, v, :] += m_uv
                agg[:, u, :] += m_vu

            node_h = self.gru(
                agg.reshape(B * N, -1),
                node_h.reshape(B * N, -1)
            ).reshape(B, N, -1)

        return node_h

class AttentionPool(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.score = nn.Linear(hidden_dim, 1)

    def forward(self, node_h):
        # node_h: (B, N, H)
        attn = torch.softmax(self.score(node_h), dim=1)  # (B, N, 1)
        pooled = (node_h * attn).sum(dim=1)              # (B, H)
        return pooled

class ClassicalFDIABinary(nn.Module):
    def __init__(self, n_buses, edge_list_1based):
        super().__init__()
        self.tcn_node = SharedTCN(T_OUT)
        self.tcn_edge = SharedTCN(T_OUT)

        self.gnn = SimpleGNN(
            n_buses=n_buses,
            edge_list_1based=edge_list_1based,
            node_in_dim=T_OUT,
            edge_in_dim=T_OUT,
            hidden=NODE_HIDDEN,
            steps=GNN_STEPS
        )

        self.pool = AttentionPool(NODE_HIDDEN)

        self.mlp = nn.Sequential(
            nn.Linear(NODE_HIDDEN, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        self.head = nn.Linear(32, 1)

    def forward(self, node_ts, edge_ts, return_embedding=False):
        node_feat = self.tcn_node(node_ts)     # (B, N, T_OUT)
        edge_feat = self.tcn_edge(edge_ts)     # (B, E, T_OUT)
        node_h = self.gnn(node_feat, edge_feat)
        pooled = self.pool(node_h)             # (B, NODE_HIDDEN)
        embedding = self.mlp(pooled)           # (B, 32)
        logits = self.head(embedding)          # (B, 1)

        if return_embedding:
            return logits, embedding
        return logits
