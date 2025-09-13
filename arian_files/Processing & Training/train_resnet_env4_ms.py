# train_resnet_env4_ms.py
# Multi-Scale ResNet1D for Scenario-1 (Env). Works with 4 or 8 channels.
# Saves: train_curve_env4_msresnet.png, cm_env_msresnet.png,
#        report_env_msresnet.txt, msresnet1d_env.pt,
#        test_preds_env_ms.csv, file_vote_report.txt

import argparse
from pathlib import Path
import numpy as np, pandas as pd, matplotlib.pyplot as plt
import torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# ---------------- utils ----------------
def parse_args():
    ap = argparse.ArgumentParser(description="Train MS-ResNet1D on env dataset")
    ap.add_argument("--data", default="processed_v2_10s_8ch",
                    help="Folder with X_env8_unscaled.npy or X_env4_unscaled.npy, y_env.npy, meta_env.csv")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--alpha", default="1,3,3,0.6,2.5",
                    help="Class weights for focal loss (labels 1..5), comma-separated")
    ap.add_argument("--gamma", type=float, default=1.2, help="Focal loss gamma")
    return ap.parse_args()

def set_seed(s):
    np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)

def stratified_group_split_by_label(files, labels0, test_frac=0.20, val_frac=0.20, seed=42):
    """Split at file-level while preserving label balance (labels 0..4)."""
    rng = np.random.default_rng(seed)
    files = np.asarray(files); labels0 = np.asarray(labels0)
    uniq_files = np.unique(files)
    file_to_class = {}
    for f in uniq_files:
        cls = np.unique(labels0[files == f])
        assert len(cls) == 1, f"File {f} maps to multiple classes"
        file_to_class[f] = int(cls[0])

    train_files, val_files, test_files = [], [], []
    n_classes = int(labels0.max() + 1)
    for k in range(n_classes):
        flist = [f for f, c in file_to_class.items() if c == k]
        rng.shuffle(flist); n = len(flist)
        if n == 0: continue
        n_test = max(1, int(round(n*test_frac))) if n >= 3 else 1 if n >= 2 else 0
        n_val  = max(1, int(round(n*val_frac)))  if n >= 3 else (1 if n >= 3 - n_test else 0)
        while n_test + n_val >= n and n > 1:
            if n_val > 1: n_val -= 1
            elif n_test > 1: n_test -= 1
            else: break
        test_sel  = flist[:n_test]
        val_sel   = flist[n_test:n_test+n_val]
        train_sel = flist[n_test+n_val:]
        if len(train_sel) == 0 and len(val_sel) > 0:
            train_sel.append(val_sel.pop())
        test_files.extend(test_sel); val_files.extend(val_sel); train_files.extend(train_sel)

    mask_train = np.isin(files, train_files)
    mask_val   = np.isin(files, val_files)
    mask_test  = np.isin(files, test_files)
    return np.flatnonzero(mask_train), np.flatnonzero(mask_val), np.flatnonzero(mask_test)

def fit_channel_standardizer(X):
    mean = X.mean(axis=(0,2), keepdims=True)
    std  = X.std(axis=(0,2), keepdims=True) + 1e-8
    return mean.astype("float32"), std.astype("float32")

def apply_channel_standardizer(X, mean, std):
    return (X - mean) / std

def print_counts(name, y):
    print(f"{name} counts:\n{pd.Series(y).value_counts().sort_index().to_string()}\n")

# ---------------- model ----------------
class BasicBlock(nn.Module):
    def __init__(self, in_ch, out_ch, k, stride=1):
        super().__init__()
        pad = k // 2
        self.conv1 = nn.Conv1d(in_ch, out_ch, k, stride=stride, padding=pad, bias=False)
        self.bn1   = nn.BatchNorm1d(out_ch)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(out_ch, out_ch, k, stride=1, padding=pad, bias=False)
        self.bn2   = nn.BatchNorm1d(out_ch)
        self.down  = None
        if stride != 1 or in_ch != out_ch:
            self.down = nn.Sequential(nn.Conv1d(in_ch, out_ch, 1, stride=stride, bias=False),
                                      nn.BatchNorm1d(out_ch))
    def forward(self, x):
        idt = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            idt = self.down(idt)
        out += idt
        return self.relu(out)

class MSResNet1D(nn.Module):
    """Small multi-scale ResNet: parallel kernels capture short/medium/long patterns."""
    def __init__(self, n_in=8, n_classes=5):
        super().__init__()
        # stem
        self.stem = nn.Sequential(
            nn.Conv1d(n_in, 64, 7, padding=3, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True)
        )
        # multi-scale stage
        self.b1_3 = BasicBlock(64, 64, 3, stride=1)
        self.b1_5 = BasicBlock(64, 64, 5, stride=1)
        self.b1_7 = BasicBlock(64, 64, 7, stride=1)
        self.merger = nn.Conv1d(64*3, 128, 1, bias=False)

        # deeper stage
        self.b2 = BasicBlock(128, 128, 3, stride=2)  # downsample T/2
        self.b3 = BasicBlock(128, 256, 3, stride=2)  # downsample T/4
        self.pool = nn.AdaptiveAvgPool1d(1)

        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(128, n_classes)
        )

    def forward(self, x):
        z = self.stem(x)                    # (B,64,T)
        z3 = self.b1_3(z)
        z5 = self.b1_5(z)
        z7 = self.b1_7(z)
        zc = torch.cat([z3, z5, z7], dim=1) # (B,64*3,T)
        zc = self.merger(zc)                # (B,128,T)
        zc = self.b2(zc)                    # (B,128,T/2)
        zc = self.b3(zc)                    # (B,256,T/4)
        zc = self.pool(zc)                  # (B,256,1)
        return self.cls(zc)                 # (B,5)

# ---------------- loss ----------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=1.2, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ce = nn.CrossEntropyLoss(weight=alpha, reduction='none')
    def forward(self, logits, target):
        ce = self.ce(logits, target)    # (B,)
        pt = torch.exp(-ce)
        loss = (1 - pt)**self.gamma * ce
        return loss.mean() if self.reduction == 'mean' else loss.sum()

# ---------------- main ----------------
def main():
    args = parse_args(); set_seed(args.seed)
    P = Path(args.data)

    # Load X (prefer 8-ch, fallback 4-ch)
    X8 = P/"X_env8_unscaled.npy"; X4 = P/"X_env4_unscaled.npy"
    if X8.exists():
        X = np.load(X8).astype("float32")          # (N,8,T)
    elif X4.exists():
        X = np.load(X4).astype("float32")          # (N,4,T)
    else:
        raise FileNotFoundError(f"Missing {X8.name} or {X4.name} in {P.resolve()}")

    y = np.load(P/"y_env.npy").astype("int64")     # labels 1..5
    meta = pd.read_csv(P/"meta_env.csv")
    y0 = y - 1                                     # 0..4
    files = meta["file"].to_numpy()

    # grouped stratified split
    tr_idx, va_idx, te_idx = stratified_group_split_by_label(files, y0, seed=args.seed)
    print_counts("TRAIN", y[tr_idx]); print_counts("VAL", y[va_idx]); print_counts("TEST", y[te_idx])

    # channel standardization (TRAIN only)
    mean, std = fit_channel_standardizer(X[tr_idx])
    Xtr = apply_channel_standardizer(X[tr_idx], mean, std)
    Xva = apply_channel_standardizer(X[va_idx], mean, std)
    Xte = apply_channel_standardizer(X[te_idx], mean, std)
    np.savez(P/"env_channel_standardizer_ms.npz", mean=mean, std=std)

    # DataLoaders
    train_loader = DataLoader(TensorDataset(torch.tensor(Xtr), torch.tensor(y0[tr_idx])),
                              batch_size=args.batch, shuffle=True, num_workers=0)
    val_loader   = DataLoader(TensorDataset(torch.tensor(Xva), torch.tensor(y0[va_idx])),
                              batch_size=256, shuffle=False, num_workers=0)
    test_loader  = DataLoader(TensorDataset(torch.tensor(Xte), torch.tensor(y0[te_idx])),
                              batch_size=256, shuffle=False, num_workers=0)

    # Model / Optim / Loss
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_in = X.shape[1]
    model = MSResNet1D(n_in=n_in, n_classes=5).to(device)
    opt   = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    sched = optim.lr_scheduler.ReduceLROnPlateau(opt, mode='max', factor=0.5, patience=4)

    alpha_vals = np.array([float(a) for a in args.alpha.split(",")], dtype=np.float32)
    if alpha_vals.size != 5:
        raise ValueError("--alpha must have 5 comma-separated values")
    alpha = torch.tensor(alpha_vals, dtype=torch.float32, device=device)
    crit  = FocalLoss(alpha=alpha, gamma=args.gamma)

    # Train loop
    best, best_state, hist = 0.0, None, {"tr":[], "va":[]}
    no_imp = 0
    for ep in range(1, args.epochs+1):
        model.train(); corr=tot=0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = crit(logits, yb)
            loss.backward(); opt.step()
            corr += (logits.argmax(1) == yb).sum().item(); tot += len(yb)
        tr_acc = corr/tot

        model.eval(); corr=tot=0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                lg = model(xb)
                corr += (lg.argmax(1)==yb).sum().item(); tot += len(yb)
        va_acc = corr/tot
        sched.step(va_acc)
        hist["tr"].append(tr_acc); hist["va"].append(va_acc)
        print(f"Epoch {ep:02d} | train {tr_acc:.3f} | val {va_acc:.3f}")

        if va_acc > best + 1e-4:
            best = va_acc
            best_state = {k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
            no_imp = 0
        else:
            no_imp += 1
            if no_imp >= args.patience:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Curves
    plt.figure()
    plt.plot(hist["tr"], label="train"); plt.plot(hist["va"], label="val")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend()
    plt.title("MS-ResNet1D — Scenario 1 (4/8ch)")
    plt.tight_layout(); plt.savefig(P/"train_curve_env4_msresnet.png", dpi=150); plt.close()

    # Test (per-frame)
    model.eval(); y_true=[]; y_pred=[]
    with torch.no_grad():
        for xb, yb in test_loader:
            lg = model(xb.to(device)).cpu().numpy()
            y_pred.append(lg.argmax(1)); y_true.append(yb.numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)

    cm = confusion_matrix(y_true, y_pred, labels=list(range(5)))
    disp = ConfusionMatrixDisplay(cm, display_labels=[1,2,3,4,5])
    fig, ax = plt.subplots(figsize=(7,6))
    disp.plot(ax=ax, values_format='d', colorbar=True)
    ax.set_title("Scenario1_Env — MS-ResNet1D (grouped & stratified)")
    fig.tight_layout(); fig.savefig(P/"cm_env_msresnet.png", dpi=150); plt.close(fig)

    rep = classification_report(y_true, y_pred, labels=list(range(5)),
                                target_names=[str(i) for i in [1,2,3,4,5]])
    (P/"report_env_msresnet.txt").write_text(rep)
    torch.save(model.state_dict(), P/"msresnet1d_env.pt")

    # ---- Per-file majority vote (NEW) ----
    meta_test = meta.iloc[te_idx].reset_index(drop=True)
    pred_df = pd.DataFrame({
        "file": meta_test["file"],
        "y_true": y_true + 1,          # back to 1..5
        "y_pred": y_pred + 1
    })
    pred_df.to_csv(P / "test_preds_env_ms.csv", index=False)

    vote = (pred_df.groupby("file")
                    .agg(y_true=("y_true", lambda s: s.mode().iat[0]),
                         y_pred=("y_pred", lambda s: s.value_counts().idxmax()),
                         n=("y_pred", "size"))
                    .reset_index())
    acc_file = (vote["y_true"] == vote["y_pred"]).mean()
    txt = [f"Per-file majority-vote accuracy: {acc_file:.3f}", "", vote.to_string(index=False)]
    (P/"file_vote_report.txt").write_text("\n".join(txt))
    print(f"[Per-file] accuracy: {acc_file:.3f}  (see test_preds_env_ms.csv & file_vote_report.txt)")
    print("Saved outputs to:", P.resolve())

if __name__ == "__main__":
    main()
