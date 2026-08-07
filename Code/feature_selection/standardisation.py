train_mean = X_train.mean()
train_std = X_train.std().replace(0, 1.0)

X_train_scaled = (X_train - train_mean) / (train_std + 1e-8)
X_val_scaled   = (X_val   - train_mean) / (train_std + 1e-8)
X_full_scaled  = (X       - train_mean) / (train_std + 1e-8)
