# =========================
# Metrics
# =========================
@torch.no_grad()
def binary_f1_from_logits(logits, y, threshold=0.5):
    probs = torch.sigmoid(logits).squeeze(1)
    preds = (probs >= threshold).long()
    y_flat = y.squeeze(1).long() if y.dim() == 2 else y.long()

    tp = ((preds == 1) & (y_flat == 1)).sum().float()
    fp = ((preds == 1) & (y_flat == 0)).sum().float()
    fn = ((preds == 0) & (y_flat == 1)).sum().float()

    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)
    f1 = 2 * precision * recall / (precision + recall + 1e-9)
    return f1.item()

# =========================
# Train / eval
# =========================
def train_epoch_binary(model, loader, opt, loss_fn):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for node_ts, edge_ts, y in tqdm(loader, leave=False):
        node_ts = node_ts.to(DEVICE)
        edge_ts = edge_ts.to(DEVICE)
        y = y.to(DEVICE).float().unsqueeze(1)

        opt.zero_grad()
        logits = model(node_ts, edge_ts)

        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        total_loss += loss.item() * y.size(0)
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total

@torch.no_grad()
def eval_epoch_binary(model, loader, loss_fn, threshold=0.5):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_logits = []
    all_y = []

    for node_ts, edge_ts, y in tqdm(loader, leave=False):
        node_ts = node_ts.to(DEVICE)
        edge_ts = edge_ts.to(DEVICE)
        y = y.to(DEVICE).float().unsqueeze(1)

        logits = model(node_ts, edge_ts)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()

        total_loss += loss.item() * y.size(0)
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

        all_logits.append(logits.detach().cpu())
        all_y.append(y.detach().cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)
    f1 = binary_f1_from_logits(all_logits, all_y, threshold=threshold)

    return total_loss / total, correct / total, f1

# =========================
# Data
# =========================
train_ds = SelectedGraphFDIADataset(TRAIN_CSV_PATH, EDGE_LIST_1BASED, N_BUSES, LABEL_COL)
val_ds   = SelectedGraphFDIADataset(VAL_CSV_PATH, EDGE_LIST_1BASED, N_BUSES, LABEL_COL)

train_labels = train_ds.df[LABEL_COL].tolist()
class_counts = Counter(train_labels)
print("Class distribution:", class_counts)

class_weights = {c: 1.0 / class_counts[c] for c in class_counts}
sample_weights = [class_weights[label] for label in train_labels]
sample_weights = torch.DoubleTensor(sample_weights)

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)

train_loader = DataLoader(
    train_ds,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    collate_fn=collate_batch
)

val_loader = DataLoader(
    val_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=collate_batch
)


# =========================
# Train
# =========================
model = ClassicalFDIABinary(N_BUSES, EDGE_LIST_1BASED).to(DEVICE)
loss_fn = nn.BCEWithLogitsLoss()
opt = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode="max", factor=0.5, patience=2
)

best_f1 = -1.0
epochs_no_improve = 0

for ep in range(1, EPOCHS + 1):
    tr_loss, tr_acc = train_epoch_binary(model, train_loader, opt, loss_fn)
    va_loss, va_acc, va_f1 = eval_epoch_binary(model, val_loader, loss_fn, threshold=THRESH)

    scheduler.step(va_f1)

    print(f"Epoch {ep:02d} | train loss {tr_loss:.4f} acc {tr_acc:.3f} | "
          f"val loss {va_loss:.4f} acc {va_acc:.3f} F1 {va_f1:.3f}")

    if va_f1 > best_f1 + MIN_DELTA:
        best_f1 = va_f1
        epochs_no_improve = 0
        torch.save(model.state_dict(), BEST_PATH)
        print(f"  ✓ New best F1: {best_f1:.4f}")
    else:
        epochs_no_improve += 1
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered.")
            break

print("Best validation F1:", best_f1)

# =========================
# Check learned embedding spread
# =========================
model.load_state_dict(torch.load(BEST_PATH, map_location=DEVICE))
model.eval()

embeds = []
labels = []

with torch.no_grad():
    for node_ts, edge_ts, y in val_loader:
        node_ts = node_ts.to(DEVICE)
        edge_ts = edge_ts.to(DEVICE)
        logits, embedding = model(node_ts, edge_ts, return_embedding=True)
        embeds.append(embedding.cpu())
        labels.append(y)

embeds = torch.cat(embeds, dim=0)
labels = torch.cat(labels, dim=0)

print("overall std:", embeds.std(dim=0).mean().item())
print("class 0 mean norm:", embeds[labels == 0].mean(dim=0).norm().item())
print("class 1 mean norm:", embeds[labels == 1].mean(dim=0).norm().item())
print("distance between class means:",
      (embeds[labels == 0].mean(dim=0) - embeds[labels == 1].mean(dim=0)).norm().item())
