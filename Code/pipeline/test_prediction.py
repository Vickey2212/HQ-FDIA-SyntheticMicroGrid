@torch.no_grad()
def get_test_predictions(model, loader, threshold=0.45):

    model.eval()

    all_probs = []
    all_preds = []
    all_y = []

    for node_ts, edge_ts, y in loader:

        node_ts = node_ts.to(DEVICE)
        edge_ts = edge_ts.to(DEVICE)

        logits = model(node_ts, edge_ts)

        probs = torch.sigmoid(logits).squeeze(1).cpu()
        preds = (probs >= threshold).long()

        all_probs.append(probs)
        all_preds.append(preds)
        all_y.append(y.cpu())

    all_probs = torch.cat(all_probs).numpy()
    all_preds = torch.cat(all_preds).numpy()
    all_y = torch.cat(all_y).numpy()

    return all_probs, all_preds, all_y
