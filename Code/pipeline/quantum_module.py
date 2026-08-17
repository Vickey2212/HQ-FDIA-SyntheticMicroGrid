class QuantumResidualBinary(nn.Module):
    def __init__(self, base_model, n_qubits=6, q_layers=3):
        super().__init__()

        self.base = base_model   # trained classical model

        # small projection into quantum space
        self.to_quantum = nn.Sequential(
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, n_qubits),
            nn.Tanh()
        )

        # quantum module
        self.quantum = QuantumReuploading(
            n_qubits=n_qubits,
            q_layers=q_layers
        )

        self.q_head = nn.Sequential(
            nn.Linear(n_qubits, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU(),
            nn.Linear(8, 1)
        )

        # fusion weight
        self.alpha = nn.Parameter(torch.tensor(0.05))

    def forward(self, node_ts, edge_ts):

        # ---- classical forward ----
        with torch.no_grad():
            logits_classical, embedding = self.base(node_ts, edge_ts, return_embedding=True)

        # ---- quantum branch ----
        q_input = self.to_quantum(embedding)
        q_feat = self.quantum(q_input)
        q_logit = self.q_head(q_feat)

        # ---- residual fusion ----
        final_logit = logits_classical + self.alpha * q_logit

        return final_logit

  class QuantumReuploading(nn.Module):
    def __init__(self, n_qubits=6, q_layers=3):
        super().__init__()
        self.n_qubits = n_qubits
        self.q_layers = q_layers

        dev = qml.device("default.qubit", wires=n_qubits)
        self.weights = nn.Parameter(0.01 * torch.randn(q_layers, n_qubits, 2))

        def get_entanglement_pairs(n_qubits, layer):
            if n_qubits == 4:
                return [(0, 1), (2, 3)] if (layer % 2 == 0) else [(1, 2), (3, 0)]
            return [(i, (i + 1) % n_qubits) for i in range(n_qubits)]

        @qml.qnode(dev, interface="torch", diff_method="parameter-shift")
        def qnode(inputs, weights):
            d = inputs.shape[0]

            for l in range(q_layers):
                for q in range(n_qubits):
                    idx = (l * n_qubits + q) % d
                    qml.RY(inputs[idx], wires=q)

                for q in range(n_qubits):
                    qml.RY(weights[l, q, 0], wires=q)
                    qml.RZ(weights[l, q, 1], wires=q)

                pairs = get_entanglement_pairs(n_qubits, l)
                for a, b in pairs:
                    qml.CNOT(wires=[a, b])

            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.qnode = qnode

    def forward(self, x):
        outputs = []
        for i in range(x.shape[0]):
            qi = self.qnode(x[i], self.weights)
            qi = torch.stack(qi)
            outputs.append(qi)
        return torch.stack(outputs, dim=0).to(dtype=x.dtype, device=x.device)
