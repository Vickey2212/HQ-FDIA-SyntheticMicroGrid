model = ClassicalFDIABinary(N_BUSES, EDGE_LIST_1BASED).to(DEVICE)
model.load_state_dict(torch.load(OUTPUT_DIR / "best_classical_end2end_binary.pt", map_location=DEVICE))
model.eval()

for p in model.parameters():
    p.requires_grad = False
