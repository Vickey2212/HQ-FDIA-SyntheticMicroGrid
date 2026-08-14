# ============================================================
# SETTINGS
# ============================================================

N_RUNS = 5
BASE_SEED = 7
EPOCHS = 10


RESULTS = []

# ============================================================
# RANDOM SEED
# ============================================================

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ============================================================
# BUILD MODEL
# ============================================================

def build_model(config):

    # FULL MODEL
    if config == "FULL":
        net = QuantumResidualBinary(
            model,
            n_qubits=4,
            q_layers=3
        ).to(DEVICE)

    # ONLY CLASSICAL
    elif config == "NO_QUANTUM":

        net = copy.deepcopy(model).to(DEVICE)

        # unfreeze all parameters
        for p in net.parameters():
            p.requires_grad = True

    # NO RESIDUAL
    elif config == "NO_RESIDUAL":

        class NoResidual(QuantumResidualBinary):

            def forward(self, node_ts, edge_ts):

                with torch.no_grad():
                    _, embedding = self.base(
                        node_ts,
                        edge_ts,
                        return_embedding=True
                    )

                q_input = self.to_quantum(embedding)
                q_feat = self.quantum(q_input)
                q_logit = self.q_head(q_feat)

                return q_logit

        net = NoResidual(
            model,
            n_qubits=4,
            q_layers=3
        ).to(DEVICE)

    # NO ATTENTION
    elif config == "NO_ATTENTION":

        class NoAttention(ClassicalFDIABinary):

            def forward(self, node_ts, edge_ts, return_embedding=False):

                node_feat = self.tcn_node(node_ts)
                edge_feat = self.tcn_edge(edge_ts)

                node_h = self.gnn(node_feat, edge_feat)

                pooled = node_h.mean(dim=1)

                embedding = self.mlp(pooled)
                logits = self.head(embedding)

                if return_embedding:
                    return logits, embedding

                return logits

        base = NoAttention(
            N_BUSES,
            EDGE_LIST_1BASED
        ).to(DEVICE)

        base.load_state_dict(
            model.state_dict(),
            strict=False
        )

        net = QuantumResidualBinary(
            base,
            n_qubits=4,
            q_layers=3
        ).to(DEVICE)

    # NO GNN
    elif config == "NO_GNN":

        class NoGNN(ClassicalFDIABinary):

            def __init__(self, n_buses, edge_list_1based):
                super().__init__(n_buses, edge_list_1based)

                self.feature_projection = nn.Sequential(
                    nn.Linear(T_OUT, NODE_HIDDEN),
                    nn.ReLU()
                )

            def forward(self, node_ts, edge_ts, return_embedding=False):

                # TCN
                node_feat = self.tcn_node(node_ts)

                # Remove graph learning
                pooled = node_feat.mean(dim=1)          

                # Recover expected dimension
                pooled = self.feature_projection(pooled)  

                embedding = self.mlp(pooled)

                logits = self.head(embedding)

                if return_embedding:
                    return logits, embedding

                return logits

        base = NoGNN(
            N_BUSES,
            EDGE_LIST_1BASED
        ).to(DEVICE)

        net = QuantumResidualBinary(
            base,
            n_qubits=4,
            q_layers=3
        ).to(DEVICE)

    # NO TCN
    elif config == "NO_TCN":

        class NoTCN(ClassicalFDIABinary):

            def __init__(self, n_buses, edge_list_1based):
                super().__init__(n_buses, edge_list_1based)

               
                self.node_projection = nn.Sequential(
                    nn.Linear(train_ds.T, T_OUT),
                    nn.ReLU()
                )

                self.edge_projection = nn.Sequential(
                    nn.Linear(train_ds.T, T_OUT),
                    nn.ReLU()
                )

            def forward(self, node_ts, edge_ts, return_embedding=False):

                # Skip TCN
                node_feat = self.node_projection(node_ts)
                edge_feat = self.edge_projection(edge_ts)

                # Keep GNN
                node_h = self.gnn(node_feat, edge_feat)

                pooled = self.pool(node_h)

                embedding = self.mlp(pooled)

                logits = self.head(embedding)

                if return_embedding:
                    return logits, embedding

                return logits

        base = NoTCN(
            N_BUSES,
            EDGE_LIST_1BASED
        ).to(DEVICE)

        net = QuantumResidualBinary(
            base,
            n_qubits=4,
            q_layers=3
        ).to(DEVICE)

    return net

# ============================================================
# RUN SINGLE EXPERIMENT
# ============================================================

def recall_at_fpr(y_true, y_score, target_fpr):
    
    fpr, tpr, thresholds = roc_curve(
        y_true,
        y_score,
        drop_intermediate=False
    )
 
    valid = fpr <= target_fpr
 
    if not np.any(valid):
        return 0.0
 
    return np.max(tpr[valid])
    
def compute_binary_metrics(y_true, y_pred, y_prob=None):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if y_prob is not None:
        try:
            auc = roc_auc_score(
                y_true,
                y_prob
            )
 
            recall_fpr_1 = recall_at_fpr(
                y_true,
                y_prob,
                0.01        # 1%
            )
 
            recall_fpr_5 = recall_at_fpr(
                y_true,
                y_prob,
                0.05        # 5%
            )
 
        except ValueError:
 
            auc = np.nan
            recall_fpr_1 = np.nan
            recall_fpr_5 = np.nan
    else:
        auc = np.nan
        recall_fpr_1 = np.nan
        recall_fpr_5 = np.nan

    tnr = tn / (tn + fp) if (tn + fp) > 0 else 0

    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    return {
        "ACC": acc,
        "PREC": prec,
        "REC": rec,
        "F1": f1,
        "AUC": auc,
        "TNR": tnr,
        "FNR": fnr,
        "TN": tn,
        "FP": fp,
        "FN": fn,
        "TP": tp,
        "Recall@1%FPR": recall_fpr_1,
        "Recall@5%FPR": recall_fpr_5
    }

def run_one_experiment(cfg, run):

    seed = BASE_SEED + run
    set_seed(seed)

    net = build_model(cfg)

    trainable_params = [p for p in net.parameters() if p.requires_grad]

    print("Trainable tensors:", len(trainable_params))

    optimizer = torch.optim.Adam(
        trainable_params,
        lr=1e-4
    )

    loss_fn = nn.BCEWithLogitsLoss()

    # ======================================
    # EARLY STOPPING SETTINGS
    # ======================================
    PATIENCE = 5
    counter = 0

    best_f1 = -1
    best_state = None
    best_val_time = None

    history = []

    for ep in range(1, EPOCHS + 1):

        print(f"{cfg} | Run {run} | Epoch {ep}")

        # ---------------- TRAIN ----------------
        tr_loss, tr_acc = train_epoch_binary(
            net,
            train_loader,
            optimizer,
            loss_fn
        )

        # ---------------- VALIDATION ----------------
        start = time.time()

        va_loss, va_acc, va_f1 = eval_epoch_binary(
            net,
            val_loader,
            loss_fn,
            threshold=THRESH
        )

        val_time = time.time() - start

        history.append({
            "epoch": ep,
            "train_loss": tr_loss,
            "val_loss": va_loss,
            "train_acc": tr_acc,
            "val_acc": va_acc,
            "val_f1": va_f1
        })

        # ======================================
        # CHECK IMPROVEMENT
        # ======================================
        if va_f1 > best_f1:

            best_f1 = va_f1
            best_state = copy.deepcopy(net.state_dict())
            best_val_time = val_time
            counter = 0

            print(f"✓ Improved Val F1 = {best_f1:.4f}")

        else:
            counter += 1
            print(f"No improvement ({counter}/{PATIENCE})")

        # ======================================
        # EARLY STOPPING
        # ======================================
        if counter >= PATIENCE:
            print("Early stopping triggered.")
            break

    # ======================================
    # LOAD BEST MODEL
    # ======================================
    net.load_state_dict(best_state)

    # ---------------- TEST ----------------

    probs, preds, y_true = get_test_predictions(
        net,
        test_loader,
        threshold=THRESH
    )

    metrics = compute_binary_metrics(
        y_true,
        preds,
        probs
    )

    history_df = pd.DataFrame(history)

    return (
      metrics["ACC"],
      metrics["PREC"],
      metrics["REC"],
      metrics["F1"],
      metrics["AUC"],
      metrics["TNR"],
      metrics["FNR"],
      best_val_time,
      metrics["Recall@1%FPR"],
      metrics["Recall@5%FPR"],
      history_df
  )


# ============================================================
# ABLATION LIST
# ============================================================

ABLATIONS = [
    "FULL",
    "NO_QUANTUM",
    "NO_RESIDUAL",
    "NO_ATTENTION",
    "NO_GNN",
    "NO_TCN"
]

# ============================================================
# RUN ALL
# ============================================================

for cfg in ABLATIONS:

    print("\n" + "="*70)
    print("RUNNING:", cfg)
    print("="*70)

    for run in range(1, N_RUNS + 1):

        acc, prec, rec, f1, auc, tnr, fnr, val_time, history_df = run_one_experiment(cfg, run)

        # ----------------------------------------------------
        # SAVE CURVES CSV
        # ----------------------------------------------------
        history_path = os.path.join(
            SAVE_DIR,
            f"history_{cfg}_run{run}.csv"
        )

        history_df.to_csv(history_path, index=False)

        # ----------------------------------------------------
        # SAVE FINAL RESULT
        # ----------------------------------------------------
        RESULTS.append({
            "Model": cfg,
            "Run": run,
            "ACC": acc,
            "PREC": prec,
            "REC": rec,
            "F1": f1,
            "AUC": auc,
            "TNR": tnr,
            "FNR": fnr,
            "Recall@1%FPR": recall_1,
            "Recall@5%FPR": recall_5,
            "VAL_TIME": val_time
        })

# ============================================================
# FINAL RESULT CSV
# ============================================================

df = pd.DataFrame(RESULTS)

final_csv = os.path.join(
    SAVE_DIR,
    "ablation_results.csv"
)

df.to_csv(final_csv, index=False)

print(df)

# ============================================================
# T-TEST
# ============================================================

print("\n" + "="*70)
print("PAIRED T TEST")
print("="*70)

full_scores = df[df.Model=="FULL"]["F1"].values

for cfg in ABLATIONS[1:]:

    other = df[df.Model==cfg]["F1"].values

    t, p = stats.ttest_rel(full_scores, other)

    print(f"FULL vs {cfg}")
    print("t =", round(t,4))
    print("p =", round(p,6))
    print()

# ============================================================
# PLOT LOSS CURVES
# ============================================================

plt.figure(figsize=(10,6))

for model_name in ABLATIONS:

    files = glob.glob(
        os.path.join(
            SAVE_DIR,
            f"history_{model_name}_run*.csv"
        )
    )

    curves = []

    for f in files:
        temp = pd.read_csv(f)
        curves.append(temp["val_loss"].values)

    # -----------------------------------
    # HANDLE DIFFERENT LENGTHS
    # -----------------------------------
    max_len = max(len(c) for c in curves)

    padded = []

    for c in curves:
        pad = np.pad(
            c,
            (0, max_len - len(c)),
            mode='edge'
        )
        padded.append(pad)

    curves = np.array(padded)

    mean_curve = curves.mean(axis=0)
    std_curve = curves.std(axis=0)

    epochs = range(1, max_len + 1)

    plt.plot(epochs, mean_curve, label=model_name)

    plt.fill_between(
        epochs,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Loss")
plt.title("Ablation Loss Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(
    os.path.join(SAVE_DIR, "loss_curves.png"),
    dpi=600
)

plt.show()

print("All files saved in:", SAVE_DIR)

# ============================================================
# PLOT ACCURACY CURVES
# ============================================================

plt.figure(figsize=(10,6))

for model_name in ABLATIONS:

    files = glob.glob(
        os.path.join(
            SAVE_DIR,
            f"history_{model_name}_run*.csv"
        )
    )

    curves = []

    for f in files:
        temp = pd.read_csv(f)
        curves.append(temp["val_acc"].values)

    # -----------------------------------
    # HANDLE VARIABLE LENGTH CURVES
    # -----------------------------------
    max_len = max(len(c) for c in curves)

    padded_curves = []

    for c in curves:
        pad = np.pad(
            c,
            (0, max_len - len(c)),
            mode='edge'
        )
        padded_curves.append(pad)

    curves = np.vstack(padded_curves)

    mean_curve = curves.mean(axis=0)
    std_curve  = curves.std(axis=0)

    epochs = range(1, max_len + 1)

    plt.plot(epochs, mean_curve, label=model_name)

    plt.fill_between(
        epochs,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15
    )

plt.xlabel("Epoch")
plt.ylabel("Validation Accuracy")
plt.title("Ablation Accuracy Curves")
plt.legend()
plt.grid(True)
plt.tight_layout()

plt.savefig(
    os.path.join(SAVE_DIR, "accuracy_curves.png"),
    dpi=600
)

plt.show()

print("All files saved in:", SAVE_DIR)
