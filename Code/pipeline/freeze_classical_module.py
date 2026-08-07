model = ClassicalFDIABinary(N_BUSES, EDGE_LIST_1BASED).to(DEVICE)
model.load_state_dict(torch.load("G:/My Drive/PhD/Implementation/RQ1/Simulation/IEEE-30-30/best_classical_end2end_binary.pt", map_location=DEVICE))
model.eval()

for p in model.parameters():
    p.requires_grad = False
