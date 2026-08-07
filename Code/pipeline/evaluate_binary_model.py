@torch.no_grad()
def eval_embed_epoch_binary(model, loader, loss_fn, threshold=0.45):
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_logits = []
    all_y = []

    for x, y in loader:
        x = x.to(DEVICE)
        y = y.to(DEVICE).float().unsqueeze(1)

        logits = model(x)
        loss = loss_fn(logits, y)

        probs = torch.sigmoid(logits)
        preds = (probs >= threshold).long()

        total_loss += loss.item() * y.size(0)
        correct += (preds == y.long()).sum().item()
        total += y.size(0)

        all_logits.append(logits.cpu())
        all_y.append(y.cpu())

    all_logits = torch.cat(all_logits, dim=0)
    all_y = torch.cat(all_y, dim=0)

    va_f1 = binary_f1_from_logits(all_logits, all_y, threshold=threshold)
    return total_loss / total, correct / total, va_f1
