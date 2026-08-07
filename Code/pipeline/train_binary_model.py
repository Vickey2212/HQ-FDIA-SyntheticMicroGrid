def train_embed_epoch_binary(model, loader, opt, loss_fn):

    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in tqdm(loader, leave=False):

        x = x.to(DEVICE)
        # y = y.to(DEVICE).float().unsqueeze(1)  # 
        y = y.to(DEVICE)

        opt.zero_grad()

        logits = model(x)

        loss = loss_fn(logits, y)

        loss.backward()

        # gradient check (only first batch)
        for name, param in model.named_parameters():
            if param.grad is not None:
                # print("Grad check:", name, param.grad.abs().mean().item())
                break

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        opt.step()

        total_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).long()

        correct += (preds == y.long()).sum().item()
        total += y.size(0)

    return total_loss / total, correct / total
