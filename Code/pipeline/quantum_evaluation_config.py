q_model = QuantumResidualBinary(model, n_qubits=4, q_layers=3).to(DEVICE)
opt = torch.optim.Adam(
    filter(lambda p: p.requires_grad, q_model.parameters()),
    lr=1e-4
)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
   opt, mode="max", factor=0.5, patience=2
)

loss_fn = nn.BCEWithLogitsLoss()

best_f1 = -1.0
epochs_no_improve = 0

history = []

BEST_MODEL_PATH = os.path.join(SAVE_DIR, "best_binary_quantum_residual.pt")
LAST_MODEL_PATH = os.path.join(SAVE_DIR, "last_binary_quantum_residual.pt")
HISTORY_CSV = os.path.join(SAVE_DIR, "training_history.csv")
PLOT_PATH = os.path.join(SAVE_DIR, "training_curves.png")
TEST_REPORT_TXT = os.path.join(SAVE_DIR, "test_report.txt")
CM_PATH = os.path.join(SAVE_DIR, "confusion_matrix.png")
SUMMARY_JSON = os.path.join(SAVE_DIR, "summary_metrics.json")

best_val_time = None
