import os, re, json, random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Tuple
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from wsn_ml_pipeline_model.config.logger import LoggerConfigurator
from wsn_ml_pipeline_model.config.constants import TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, SCENARIO, SEEN_SPLIT, HELD_OUT_ENV, \
    HELD_OUT_NODE, BATCH_SIZE, EPOCHS, LR, SEED, INPUT_CHANNEL, MODEL_TYPE, TEST_SIZE



DEVICE         = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
logger = LoggerConfigurator.setup_logging()

# Filename format: frames_<Tx>_<Rx>_<env>.npy
NAME_RE = re.compile(r"^frames_(?P<tx>[A-Z])_(?P<rx>[A-Z])_(?P<env>[1-5])\.npy$")

def parse_name(fname: str):
    m = NAME_RE.match(fname)
    if not m:
        raise ValueError(f"Bad filename: {fname}")
    rx  = m.group("rx")
    tx  = m.group("tx")
    env = int(m.group("env"))
    return rx, tx, env

def load_all_frames(TRAIN_INPUT_DIR: str, scenario: str):
    """
    Read all .npy files in the directory (each is (K, L, 2)), merge into (N, 2, L) and labels y.
    Scenario I: label = environment env
    Scenario II: label = receiving node rx
    """
    L_ref = None
    X_list, y_list = [], []
    meta = []
    file_infos = []
    all_rxs = set()
    all_envs = set()

    for p in sorted(Path(TRAIN_INPUT_DIR).glob("*.npy")):
        rx, tx, env = parse_name(p.name)
        file_infos.append((p, rx, tx, env))
        all_rxs.add(rx)
        all_envs.add(env)

    # Build sorted class maps
    if scenario == "I":
        env_map = {k: i for i, k in enumerate(sorted(all_envs))}
    else:
        rx_map = {k: i for i, k in enumerate(sorted(all_rxs))}

    for p, rx, tx, env in file_infos:
        arr = np.load(p)
        arr = arr[..., 1:]               # keep only RSSI and LQI
        if arr.ndim != 3 or arr.shape[-1] != 2:
            raise ValueError(f"{p.name}: expect (K,L,2), got {arr.shape}")

        K, L, C = arr.shape
        if L_ref is None: L_ref = L
        assert L == L_ref, f"All files must share same frame length. {p.name} has L={L}, ref={L_ref}"

        # Convert to (K, C, L) to fit 1D-CNN, support RSSI, LQI or both
        if INPUT_CHANNEL == "rssi":
            selected = arr[..., 0:1]
        elif INPUT_CHANNEL == "lqi":
            selected = arr[..., 1:2]
        elif INPUT_CHANNEL == "both":
            selected = arr[..., 0:2]
        else:
            raise ValueError(f"Invalid INPUT_CHANNEL: {INPUT_CHANNEL}")
        frames = np.transpose(selected.astype(np.float32), (0, 2, 1))  # (K, C, L)

        label = env_map[env] if scenario == "I" else rx_map[rx]
        X_list.append(frames)
        y_list.append(np.full((K,), label, dtype=np.int64))
        meta.append({"file": p.name, "rx": rx, "tx": tx, "env": env, "num_frames": int(K)})

    if not X_list:
        raise RuntimeError("No npy files found.")

    X = np.concatenate(X_list, axis=0)   # (N, C, L)
    y = np.concatenate(y_list, axis=0)   # (N,)
    class_map = env_map if scenario == "I" else rx_map
    return X, y, class_map, L_ref, meta

# -------------------- Dataset --------------------
class FramesDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, C, L), float32
        self.y = torch.from_numpy(y)  # (N,), int64
    def __len__(self): return self.X.shape[0]
    def __getitem__(self, i):
        return self.X[i], self.y[i]

# -------------------- Model: Simple 1D-CNN --------------------
class SimpleCNN1D(nn.Module):
    def __init__(self, in_ch=2, num_classes=5, L=100):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_ch, 32, kernel_size=7, padding=3), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=5, padding=2), nn.ReLU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.fc = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.net(x)              # (N, 128, 1)
        x = x.squeeze(-1)            # (N, 128)
        return self.fc(x)

# -------------------- Model: ResNet1D (small) --------------------
class ResidualBlock1D(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(ch_in, ch_out, 3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm1d(ch_out)
        self.relu  = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(ch_out, ch_out, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm1d(ch_out)
        self.down  = None
        if stride != 1 or ch_in != ch_out:
            self.down = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(ch_out)
            )
    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out

class ResNet1D(nn.Module):
    def __init__(self, in_ch=2, num_classes=5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv1d(in_ch, 32, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock1D(32, 64,  stride=1)
        self.layer2 = ResidualBlock1D(64, 128, stride=2)
        self.layer3 = ResidualBlock1D(128, 128, stride=1)
        self.gap    = nn.AdaptiveAvgPool1d(1)
        self.fc     = nn.Linear(128, num_classes)
    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# -------------------- Training/Evaluation --------------------
def run_epoch(model, loader, crit, opt=None):
    is_train = opt is not None
    model.train(is_train)
    total_loss, y_true, y_pred = 0.0, [], []
    for xb, yb in loader:
        xb, yb = xb.to(DEVICE), yb.to(DEVICE)
        logits = model(xb)
        loss = crit(logits, yb)
        if is_train:
            opt.zero_grad(); loss.backward(); opt.step()
        total_loss += loss.item() * xb.size(0)
        y_true.append(yb.detach().cpu().numpy())
        y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
    y_true = np.concatenate(y_true); y_pred = np.concatenate(y_pred)
    return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred

def main():
    # Step 1: Load data and class labels from preprocessed .npy files
    X, y, class_map, L, meta = load_all_frames(TRAIN_INPUT_DIR, SCENARIO)
    num_classes = len(class_map)
    logger.info("========= Configuration Summary =========")
    logger.info(f"Model       : {MODEL_TYPE.upper()}")
    logger.info(f"Channel     : {INPUT_CHANNEL.upper()}")
    logger.info(f"Scenario    : {SCENARIO}")
    logger.info(f"Epochs      : {EPOCHS}")
    logger.info(f"Classes     : {num_classes}")
    logger.info(f"Input Shape : X={X.shape}, y={y.shape}, L={L}")

    # Step 2: Construct run ID and output directory for saving results
    split_name = f"Seen({TEST_SIZE})" if SEEN_SPLIT else f"Unseen({HELD_OUT_ENV if SCENARIO == 'II' else HELD_OUT_NODE})"
    tag = f"{SCENARIO}_{split_name}_{MODEL_TYPE}_{INPUT_CHANNEL}_epoch({EPOCHS})"
    run_id = datetime.now().strftime("%Y%m%d_%H%M")
    RUN_DIR = os.path.join(TRAIN_OUTPUT_DIR, f"{tag}_{run_id}")
    # os.makedirs(RUN_DIR, exist_ok=True)

    # Step 3: Create training and testing splits (Seen or Unseen strategy)
    if SEEN_SPLIT: # Seen
        Xtr, Xte, ytr, yte = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=SEED, stratify=y
        )
        logger.info(f"Seen Split  : test_size={TEST_SIZE}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
        
    else: # Unseen
        if SCENARIO == "I":
            # I: leave out one node across all environments
            assert isinstance(HELD_OUT_NODE, str)
            node_per_frame = []
            for p in sorted(Path(TRAIN_INPUT_DIR).glob("*.npy")):
                rx, tx, env = parse_name(p.name)
                arr = np.load(p)
                K = arr.shape[0]
                node_per_frame += [rx] * K
            node_per_frame = np.array(node_per_frame)
            mask_te = (node_per_frame == HELD_OUT_NODE)
            Xtr, Xte, ytr, yte = X[~mask_te], X[mask_te], y[~mask_te], y[mask_te]
            logger.info(f"Unseen Split: unseen_node={HELD_OUT_NODE}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
        elif SCENARIO == "II":
            # II: leave out one environment for each node
            env_rx_per_frame = []
            for p in sorted(Path(TRAIN_INPUT_DIR).glob("*.npy")):
                rx, tx, env = parse_name(p.name)
                arr = np.load(p)
                K = arr.shape[0]
                env_rx_per_frame += [(env, rx)] * K
            env_rx_per_frame = np.array(env_rx_per_frame, dtype=object)
            mask_te = np.array([e == HELD_OUT_ENV for e, _ in env_rx_per_frame])
            Xtr, Xte, ytr, yte = X[~mask_te], X[mask_te], y[~mask_te], y[mask_te]
            logger.info(f"Unseen Split: unseen_env={HELD_OUT_ENV}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
        else:
            raise ValueError(f"Invalid SCENARIO: {SCENARIO}")
    logger.info("==========================================\n")

    # Step 4: Wrap data into PyTorch Datasets and DataLoaders
    tr_ds = FramesDataset(Xtr, ytr)
    te_ds = FramesDataset(Xte, yte)
    tr_ld = DataLoader(tr_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    te_ld = DataLoader(te_ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Step 5: Initialize CNN or ResNet model
    if MODEL_TYPE == "cnn":
        model = SimpleCNN1D(in_ch=X.shape[1], num_classes=num_classes, L=L).to(DEVICE)
    elif MODEL_TYPE == "resnet":
        model = ResNet1D(in_ch=X.shape[1], num_classes=num_classes).to(DEVICE)
    else:
        raise ValueError(f"Invalid MODEL_TYPE: {MODEL_TYPE}")

    crit = nn.CrossEntropyLoss()
    opt  = torch.optim.Adam(model.parameters(), lr=LR)

    # Step 6: Train model over multiple epochs, track best accuracy
    best_acc, best_state = 0.0, None
    for epoch in range(1, EPOCHS+1):
        tr_loss, tr_acc, _, _ = run_epoch(model, tr_ld, crit, opt)
        te_loss, te_acc, y_true, y_pred = run_epoch(model, te_ld, crit, opt=None)
        if te_acc > best_acc:
            best_acc, best_state = te_acc, {k: v.cpu() for k, v in model.state_dict().items()}
        logger.info(f"Epoch {epoch:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                    f"test loss {te_loss:.4f} acc {te_acc:.4f}")

    # Step 7: Evaluate model on test set using accuracy, report, and confusion matrix
    if best_state is not None:
        model.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
    _, _, y_true, y_pred = run_epoch(model, te_ld, crit, opt=None)
    logger.info(f"Best test acc: {best_acc:.4f}\n")
    logger.info("Classification report:\n" + classification_report(y_true, y_pred))
    logger.info("Confusion matrix:\n" + str(confusion_matrix(y_true, y_pred)))

    # Step 8: Visualize and save confusion matrix to PNG
    os.makedirs(RUN_DIR, exist_ok=True) # ready for saving result
    def plot_confusion_matrix_matplotlib(y_true, y_pred, labels, title, normalize=True, cmap=plt.cm.Blues):
        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100  # percentage

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        vmax = np.ceil(cm.max() / 10) * 10
        im.set_clim(0, vmax)

        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               xticklabels=labels, yticklabels=labels,
               xlabel='Predicted Label', ylabel='True Label',
               title=title)

        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        fmt = ".1f"
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        filename = f"ConfMat_{tag}"
        save_path = os.path.abspath(f"{RUN_DIR}/{filename.replace(' ', '_')}.png")
        logger.info(f"Saving confusion matrix to: {save_path}")
        plt.savefig(save_path)
        plt.savefig(f"{RUN_DIR}/{filename.replace(' ', '_')}.png")
        plt.close()

    inv_class_map = {v: k for k, v in class_map.items()}
    label_names = sorted([inv_class_map[i] for i in range(len(inv_class_map))])

    plot_confusion_matrix_matplotlib(
        y_true, y_pred, labels=label_names,
        title=f"{MODEL_TYPE.upper()} {INPUT_CHANNEL.upper()}"
    )

    # Step 9: Save final model, metadata, and evaluation artifacts
    torch.save(model.state_dict(), f"{RUN_DIR}/model_{tag}.pt")
    # Ensure class_map keys are sorted alphabetically before saving
    sorted_class_map = {k: class_map[k] for k in sorted(class_map)}
    with open(f"{RUN_DIR}/meta_{tag}.json", "w") as f:
        json.dump({"scenario": SCENARIO, "class_map": sorted_class_map, 
                   "model": MODEL_TYPE, "channel": INPUT_CHANNEL, "epochs": EPOCHS, "L": L,
                   "seen_split": split_name, 
                   "X_shape": X.shape, "y_len": int(y.shape[0])}, f, indent=2)

if __name__ == "__main__":
    main()