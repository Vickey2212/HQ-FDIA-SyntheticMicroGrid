@torch.no_grad()
def binary_f1_from_logits(logits, y, threshold=0.45):

    # Convert logits to probabilities
    probs = torch.sigmoid(logits).squeeze(1)

    # Convert probabilities to binary predictions
    preds = (probs >= threshold).long()

    # Ensure ground-truth labels are one-dimensional
    y_flat = y.squeeze(1).long() if y.dim() == 2 else y.long()

    # Calculate confusion-matrix components
    tp = ((preds == 1) & (y_flat == 1)).sum().float()
    fp = ((preds == 1) & (y_flat == 0)).sum().float()
    fn = ((preds == 0) & (y_flat == 1)).sum().float()

    # Precision and recall
    precision = tp / (tp + fp + 1e-9)
    recall = tp / (tp + fn + 1e-9)

    # F1 score
    f1 = 2 * precision * recall / (precision + recall + 1e-9)

    return f1.item()
