@torch.no_grad()
def binary_f1_from_logits(logits, y, threshold=0.45):
    """
    Compute the binary F1 score from model logits.

    Parameters
    ----------
    logits : torch.Tensor
        Raw model outputs with shape (N, 1).
    y : torch.Tensor
        Ground-truth binary labels with shape (N,) or (N, 1).
    threshold : float, default=0.45
        Probability threshold used to convert sigmoid probabilities
        into binary predictions.

    Returns
    -------
    float
        Binary F1 score.
    """

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
