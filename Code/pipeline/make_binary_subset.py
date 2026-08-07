def make_binary_subset(dataset, n0=5000, n1=5000, seed=7):
    random.seed(seed)
    idx0, idx1 = [], []

    for i in range(len(dataset)):
        _, y = dataset[i]
        if y.item() == 0:
            idx0.append(i)
        else:
            idx1.append(i)

    random.shuffle(idx0)
    random.shuffle(idx1)

    selected = idx0[:n0] + idx1[:n1]
    random.shuffle(selected)

    return Subset(dataset, selected)
