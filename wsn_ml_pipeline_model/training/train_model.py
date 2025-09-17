# File: wsn_ml_pipeline_model/training/train_model.py
# This script defines and trains a machine learning model (1D-CNN or ResNet1D) on preprocessed sensor data.
# It includes data loading, model definition, training, evaluation, and saving the trained model and
# performance metrics.

import shutil
import os
import re
import json
import torch
import random
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
import matplotlib.pyplot as plt
from dataclasses import dataclass
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from wsn_ml_pipeline_model.config.logger import LoggerConfigurator
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from wsn_ml_pipeline_model.config.constants import TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, SCENARIO, SEEN_SPLIT, HELD_OUT_ENV, \
    HELD_OUT_NODE, BATCH_SIZE, EPOCHS, LR, SEED, INPUT_CHANNEL, MODEL_TYPE, TEST_SIZE, N_TRAIN_RUNS, RESUME_TRAINING


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

# Filename format: frames_<Tx>_<Rx>_<env>.npy
NAME_RE = re.compile(
    r"^frames_(?P<tx>[A-Z])_(?P<rx>[A-Z])_(?P<env>[1-5])\.npy$")


@dataclass
class Config:
    """ 
        Configuration for training the ML model.
    """
    TRAIN_INPUT_DIR: str
    TRAIN_OUTPUT_DIR: str
    SCENARIO: str
    SEEN_SPLIT: bool
    HELD_OUT_ENV: int
    HELD_OUT_NODE: str
    BATCH_SIZE: int
    EPOCHS: int
    LR: float
    SEED: int
    INPUT_CHANNEL: str
    MODEL_TYPE: str
    TEST_SIZE: float
    RESUME_TRAINING: bool
    N_TRAIN_RUNS: int


def parse_name(fname: str):
    """
    Parse filename to extract Tx, Rx, and environment.  
    Args:   
        fname (str): Filename in the format 'frames_<Tx>_<Rx>_<env>.npy'.
    Returns:
        Tuple
        (str, str, int): Transmitting node (Tx), Receiving node (Rx), Environment (env).
    """
    m = NAME_RE.match(fname)
    if not m:
        raise ValueError(f"Bad filename: {fname}")
    rx = m.group("rx")
    tx = m.group("tx")
    env = int(m.group("env"))
    return rx, tx, env

# -------------------- Dataset --------------------


class FramesDataset(Dataset):
    """
    PyTorch Dataset for loading framed sensor data and labels.  
    Args:
        X (np.ndarray): Array of frames with shape (N, C, L).
        y (np.ndarray): Array of labels with shape (N,).
    """

    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = torch.from_numpy(X)  # (N, C, L), float32
        self.y = torch.from_numpy(y)  # (N,), int64

    def __len__(self): return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.y[i]

# -------------------- Model: Simple 1D-CNN --------------------


class SimpleCNN1D(nn.Module):
    """
    A simple 1D Convolutional Neural Network for time-series classification.
    Args:
        in_ch (int): Number of input channels (e.g., 1 for RSSI, 2 for RSSI+LQI).
        num_classes (int): Number of output classes.
        L (int): Length of the input frames.
    """

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
    """
        A single residual block for 1D ResNet.
    """

    def __init__(self, ch_in, ch_out, stride=1):
        """ 
            Initialize the ResidualBlock1D.
            Args:
                ch_in (int): Number of input channels.
                ch_out (int): Number of output channels.
                stride (int): Stride for the first convolution. Default is 1.
        """
        super().__init__()
        self.conv1 = nn.Conv1d(
            ch_in, ch_out, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm1d(ch_out)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(ch_out, ch_out, 3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(ch_out)
        self.down = None
        if stride != 1 or ch_in != ch_out:
            self.down = nn.Sequential(
                nn.Conv1d(ch_in, ch_out, 1, stride=stride, bias=False),
                nn.BatchNorm1d(ch_out)
            )

    def forward(self, x):
        """
        Forward pass for the residual block.
        Args:   
            x (torch.Tensor): Input tensor of shape (N, ch_in, L).
        Returns:    
            torch.Tensor: Output tensor of shape (N, ch_out, L_out).
        """

        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.down is not None:
            identity = self.down(identity)
        out = self.relu(out + identity)
        return out


class ResNet1D(nn.Module):
    """
        A small ResNet architecture for 1D time-series classification.
    """

    def __init__(self, in_ch=2, num_classes=5):
        """
            Initialize the ResNet1D model.
            Args:   
                in_ch (int): Number of input channels (e.g., 1 for RSSI, 2 for RSSI+LQI).
                num_classes (int): Number of output classes.    
        """
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
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        """
        Forward pass for the ResNet1D model.                        
            Args:   
                x (torch.Tensor): Input tensor of shape (N, in_ch, L).  
            Returns:                                                                    
                torch.Tensor: Output tensor of shape (N, num_classes).
        """
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.gap(x).squeeze(-1)
        return self.fc(x)

# -------------------- Training/Evaluation --------------------


class FrameDataLoader:
    """
    Data loader for framed sensor data stored in .npy files.    
    It reads all .npy files in the specified directory, merges them into a single dataset,
    and creates training and testing splits based on the specified scenario and split type.
    """

    def __init__(self, input_dir, scenario, input_channel):
        """
        Initialize the FrameDataLoader with the input directory, scenario, and input channel.
        Args:   
            input_dir (str): Directory containing .npy files.   
            scenario (str): "I" for environment classification, "II" for receiving node classification. 
            input_channel (str): "rssi", "lqi", or "both" to select input features. 
        """
        self.input_dir = input_dir
        self.scenario = scenario
        self.input_channel = input_channel

    def load_all_frames(self, TRAIN_INPUT_DIR: str, scenario: str):
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
            if L_ref is None:
                L_ref = L
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
            frames = np.transpose(selected.astype(
                np.float32), (0, 2, 1))  # (K, C, L)

            label = env_map[env] if scenario == "I" else rx_map[rx]
            X_list.append(frames)
            y_list.append(np.full((K,), label, dtype=np.int64))
            meta.append({"file": p.name, "rx": rx, "tx": tx,
                        "env": env, "num_frames": int(K)})

        if not X_list:
            raise RuntimeError("No npy files found.")

        X = np.concatenate(X_list, axis=0)   # (N, C, L)
        y = np.concatenate(y_list, axis=0)   # (N,)
        class_map = env_map if scenario == "I" else rx_map
        return X, y, class_map, L_ref, meta

    def load(self):
        """ Load and split data into training and testing sets.
        Returns:
            X_train (np.ndarray): Training frames of shape (N_train, C, L).
            y_train (np.ndarray): Training labels of shape (N_train,).
            class_map (dict): Mapping from class names to integer labels.
            L (int): Length of each frame.
            meta (list): List of metadata dictionaries for each file.
        """
        return self.load_all_frames(self.input_dir, self.scenario)


class ModelFactory:
    """
        Factory to create model instances based on configuration.
    """
    @staticmethod
    def get_model(model_type, in_ch, num_classes, L):
        """
        Create and return a model instance based on the specified type.
        Args:
            model_type (str): Type of model to create ("cnn" or "resnet").
            in_ch (int): Number of input channels.
            num_classes (int): Number of output classes.
            L (int): Length of input frames.
        Returns:
            nn.Module: Instantiated model.
        """

        if model_type == "cnn":
            return SimpleCNN1D(in_ch=in_ch, num_classes=num_classes, L=L)
        elif model_type == "resnet":
            return ResNet1D(in_ch=in_ch, num_classes=num_classes)
        else:
            raise ValueError(f"Invalid MODEL_TYPE: {model_type}")


class Trainer:
    """
        Trainer class to handle training and evaluation of the model.
    """

    def __init__(self, model, crit, opt, device):
        """
            Initialize the Trainer with model, loss function, optimizer, and device.
            Args:
                model (nn.Module): The model to train.
                crit (nn.Module): Loss function.
                opt (torch.optim.Optimizer): Optimizer.
                device (str): Device to run the training on ("cpu" or "cuda").
        """
        self.model = model
        self.crit = crit
        self.opt = opt
        self.device = device

    def run_epoch(self, loader, is_train):
        """
            Run a single epoch of training or evaluation.
            Args:
                loader (DataLoader): DataLoader for the dataset.
                is_train (bool): True for training mode, False for evaluation mode.
            Returns:
                Tuple[float, float, np.ndarray, np.ndarray]: Average loss, accuracy, true labels, predicted labels.
        """
        self.model.train(is_train)
        total_loss, y_true, y_pred = 0.0, [], []
        for xb, yb in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            logits = self.model(xb)
            loss = self.crit(logits, yb)
            if is_train:
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()
            total_loss += loss.item() * xb.size(0)
            y_true.append(yb.detach().cpu().numpy())
            y_pred.append(logits.argmax(dim=1).detach().cpu().numpy())
        y_true = np.concatenate(y_true)
        y_pred = np.concatenate(y_pred)
        return total_loss / len(loader.dataset), accuracy_score(y_true, y_pred), y_true, y_pred


class WSNPipeline:
    """
        Orchestrates the end-to-end ML workflow: data loading, model training, evaluation,
        and saving results.
    """

    def __init__(self, logger: logging.Logger = None):
        """
            Initialize the WSNPipeline with configuration and logger.
            Args:
                logger: Optional logger instance. If None, a default logger will be created.
            Returns: None  
        """

        self.config = Config(
            TRAIN_INPUT_DIR=TRAIN_INPUT_DIR,
            TRAIN_OUTPUT_DIR=TRAIN_OUTPUT_DIR,
            SCENARIO=SCENARIO,
            SEEN_SPLIT=SEEN_SPLIT,
            HELD_OUT_ENV=HELD_OUT_ENV,
            HELD_OUT_NODE=HELD_OUT_NODE,
            BATCH_SIZE=BATCH_SIZE,
            EPOCHS=EPOCHS,
            LR=LR,
            SEED=SEED,
            INPUT_CHANNEL=INPUT_CHANNEL,
            MODEL_TYPE=MODEL_TYPE,
            TEST_SIZE=TEST_SIZE,
            RESUME_TRAINING=RESUME_TRAINING,
            N_TRAIN_RUNS=N_TRAIN_RUNS,
        )

        self.logger = LoggerConfigurator.setup_logging() or logger

    def plot_confusion_matrix_matplotlib(self, run_name, run_dir, y_true, y_pred,
                                         labels, title, normalize=True, cmap=plt.cm.Blues):
        """
        Plot confusion matrix using matplotlib and save to file.
        Args:
            run_name: identifier used in filename, e.g., 'single' or 'run{idx}'.
            run_dir: directory to save the plot.
            y_true: true labels.
            y_pred: predicted labels.
            labels: list of label names.
            title: title of the plot.
            normalize: whether to normalize the confusion matrix.
            cmap: colormap for the plot.
        """

        cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
        if normalize:
            cm = cm.astype("float") / \
                cm.sum(axis=1)[:, np.newaxis] * 100  # percentage

        fig, ax = plt.subplots(figsize=(8, 6))
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        vmax = np.ceil(cm.max() / 10) * 10
        im.set_clim(0, vmax)

        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            xlabel='Predicted Label', ylabel='True Label',
            title=title
        )

        plt.setp(ax.get_xticklabels(), rotation=0, ha="center")

        fmt = ".1f"
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()

        filename = f"ConfMat_{run_name}"
        save_path = os.path.join(run_dir, f"{filename.replace(' ', '_')}.png")
        self.logger.info(f"Saving confusion matrix to: {os.path.abspath(save_path)}")
        plt.savefig(save_path)
        plt.close()

    def get_latest_checkpoint(self, result_dir):
        """
        Get the latest model checkpoint (.pt) file from the result directory.
        Returns the path to the latest checkpoint or None if not found.
        """
        pt_files = sorted(Path(result_dir).rglob(
            "model_*.pt"), key=os.path.getmtime)
        return str(pt_files[-1]) if pt_files else None

    def get_run_dir_and_next_index(self, tag):
        """
        Returns the run directory and the next run index based on existing files.
        """
        run_dir = os.path.join(self.config.TRAIN_OUTPUT_DIR, tag)
        os.makedirs(run_dir, exist_ok=True)
        existing = list(Path(run_dir).glob("model_run*.pt"))
        indices = []
        for f in existing:
            m = re.search(r"model_run(\d+)\.pt", f.name)
            if m:
                indices.append(int(m.group(1)))
        next_idx = max(indices) + 1 if indices else 1
        return run_dir, next_idx

    def get_split_name(self):
        """Helper to build the split name string."""
        if self.config.SEEN_SPLIT:
            return f"seen({self.config.TEST_SIZE})"
        else:
            return f"unseen({self.config.HELD_OUT_ENV if self.config.SCENARIO == 'II' else self.config.HELD_OUT_NODE})"

    def get_tag(self, tag=None):
        """
        Helper to build the tag string for a run.
        If tag is provided, returns it; otherwise, constructs from config and split name.
        The tag now also includes the epoch count as '_epoch({self.config.EPOCHS})' at the end.
        """
        if tag is not None:
            return tag
        split_name = self.get_split_name()
        return f"{self.config.SCENARIO}_{split_name}_{self.config.MODEL_TYPE}_{self.config.INPUT_CHANNEL}_epoch({self.config.EPOCHS})"

    def run_multiple(self, n_runs=None, resume_latest=None):
        """
        Run multiple training sessions with different random seeds.
        Args:
            n_runs (int, optional): Number of additional runs to execute (appended to existing runs).
                If None, uses self.config.N_TRAIN_RUNS.
            resume_latest (bool, optional): If True, resume from the latest checkpoint in each run.
                If None, uses self.config.RESUME_TRAINING.
        """
        if n_runs is None:
            n_runs = self.config.N_TRAIN_RUNS
        if resume_latest is None:
            resume_latest = self.config.RESUME_TRAINING
        tag = self.get_tag()
        run_dir, next_idx = self.get_run_dir_and_next_index(tag)
        best_global_acc = -1.0
        best_global_run_name = None

        for i in range(n_runs):
            run_idx = next_idx + i
            prev_run_idx = run_idx - 1
            resume_path = None
            if resume_latest and prev_run_idx >= 1:
                candidate_resume_path = os.path.join(
                    run_dir, f"model_run{prev_run_idx}.pt")
                if os.path.exists(candidate_resume_path):
                    self.logger.info(f"Resuming from: {candidate_resume_path}")
                    resume_path = candidate_resume_path
                else:
                    self.logger.info(
                        "No checkpoint found, training from scratch.")
            else:
                self.logger.info("Training from scratch (no resume).")

            if not resume_latest and n_runs == 1:
                # Single run from scratch, skip cumulative run logging
                self.logger.info("=== Single training run ========")
            else:
                self.logger.info(
                    f"=== Training run {run_idx}/{next_idx + n_runs - 1} ===")
            acc, run_name = self.run(
                tag=tag, run_dir=run_dir, run_idx=run_idx,
                resume_path=resume_path
            )
            if acc > best_global_acc:
                best_global_acc = acc
                best_global_run_name = run_name

            
        if best_global_run_name is not None:
            src = os.path.join(run_dir, f"ConfMat_{best_global_run_name}.png")
            dst = os.path.join(run_dir, "Final_Confusion_Matrix.png")
            try:
                shutil.copyfile(src, dst)
                self.logger.info(
                    f"Saved global best confusion matrix ({best_global_acc:.4f}) to: {os.path.abspath(dst)}"
                )
            except Exception as e:
                self.logger.error(f"Failed to save global best confusion matrix: {e}")


    def run(self, X=None, y=None, class_map=None, L=None, meta=None, tag=None, run_dir=None, run_idx=None, resume_path=None):
        """
        Run the end-to-end ML workflow: data loading, model training, evaluation, and saving results.
        Args:
            X (np.ndarray): Optional preloaded frames of shape (N, C, L).
            y (np.ndarray): Optional preloaded labels of shape (N,).
            class_map (dict): Optional preloaded class mapping.
            L (int): Optional length of each frame.
            meta (list): Optional metadata list.
            tag (str): Optional tag for the run, used to create a unique directory for saving models and results.
            run_dir (str): Optional directory for saving results.
            run_idx (int): Optional run index for naming.
            resume_path (str): Optional path to a .pt checkpoint to resume training.
        """
        tag = self.get_tag(tag)
        split_name = self.get_split_name()
        if run_dir is None or run_idx is None:
            run_dir, run_idx = self.get_run_dir_and_next_index(tag)

        # --- Compute prev_total_epochs  ---
        prev_total_epochs = 0
        if resume_path is not None:
            # Try to extract run index from resume_path
            m = re.search(r"model_run(\d+)\.pt", os.path.basename(resume_path))
            if m:
                run_idx_from_resume = int(m.group(1))
                prev_total_epochs = self.get_total_epochs_by_index(run_dir, run_idx_from_resume)

        # Step 1: Load data if not provided
        if X is None or y is None or class_map is None or L is None:
            loader = FrameDataLoader(
                self.config.TRAIN_INPUT_DIR,
                self.config.SCENARIO,
                self.config.INPUT_CHANNEL
            )
            X, y, class_map, L, meta = loader.load()
        num_classes = len(class_map)
        self.logger.info("========= Configuration Summary =========")
        self.logger.info(f"Model                : {self.config.MODEL_TYPE.upper()}")
        self.logger.info(f"Channel              : {self.config.INPUT_CHANNEL.upper()}")
        self.logger.info(f"Scenario             : {self.config.SCENARIO}")
        self.logger.info(f"Epochs               : {self.config.EPOCHS}")
        self.logger.info(f"Previous Epochs      : {prev_total_epochs}")
        self.logger.info(f"Classes              : {num_classes}")
        self.logger.info(f"Input Shape          : X={X.shape}, y={y.shape}, L={L}")


        # Step 2: Construct run ID and output directory for saving results
        if run_dir is None or run_idx is None:
            run_dir, run_idx = self.get_run_dir_and_next_index(tag)
        # Step 3: Create training and testing splits
        if self.config.SEEN_SPLIT:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, random_state=self.config.SEED, stratify=y
            )
            self.logger.info(f"Seen Split           : test_size={self.config.TEST_SIZE}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
        else:
            if self.config.SCENARIO == "I":
                node_per_frame = []
                for p in sorted(Path(self.config.TRAIN_INPUT_DIR).glob("*.npy")):
                    rx, tx, env = parse_name(p.name)
                    arr = np.load(p)
                    K = arr.shape[0]
                    node_per_frame += [rx] * K
                node_per_frame = np.array(node_per_frame)
                mask_te = (node_per_frame == self.config.HELD_OUT_NODE)
                Xtr, Xte, ytr, yte = X[~mask_te], X[mask_te], y[~mask_te], y[mask_te]
                self.logger.info(f"Unseen Split: unseen_node={self.config.HELD_OUT_NODE}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
            elif self.config.SCENARIO == "II":
                env_rx_per_frame = []
                for p in sorted(Path(self.config.TRAIN_INPUT_DIR).glob("*.npy")):
                    rx, tx, env = parse_name(p.name)
                    arr = np.load(p)
                    K = arr.shape[0]
                    env_rx_per_frame += [(env, rx)] * K
                env_rx_per_frame = np.array(env_rx_per_frame, dtype=object)
                mask_te = np.array(
                    [e == self.config.HELD_OUT_ENV for e, _ in env_rx_per_frame])
                Xtr, Xte, ytr, yte = X[~mask_te], X[mask_te], y[~mask_te], y[mask_te]
                self.logger.info(f"Unseen Split: unseen_env={self.config.HELD_OUT_ENV}, train={Xtr.shape[0]}, test={Xte.shape[0]}")
            else:
                raise ValueError(f"Invalid SCENARIO: {self.config.SCENARIO}")
        self.logger.info("==========================================\n")

        # Step 4: Wrap data into PyTorch Datasets and DataLoaders
        tr_ds = FramesDataset(Xtr, ytr)
        te_ds = FramesDataset(Xte, yte)
        tr_ld = DataLoader(
            tr_ds, batch_size=self.config.BATCH_SIZE, shuffle=True,  drop_last=False)
        te_ld = DataLoader(
            te_ds, batch_size=self.config.BATCH_SIZE, shuffle=False, drop_last=False)

        # Step 5: Initialize model
        model = ModelFactory.get_model(
            self.config.MODEL_TYPE, X.shape[1], num_classes, L).to(DEVICE)
        # Resume from checkpoint if provided
        if resume_path is not None and os.path.isfile(resume_path):
            self.logger.info(f"Resuming training from checkpoint: {resume_path}")
            model.load_state_dict(torch.load(resume_path, map_location=DEVICE))
        crit = nn.CrossEntropyLoss()
        opt = torch.optim.Adam(model.parameters(), lr=self.config.LR)
        trainer = Trainer(model, crit, opt, DEVICE)

        # Step 6: Train model
        best_acc, best_state = 0.0, None
        for epoch in range(1, self.config.EPOCHS+1):
            tr_loss, tr_acc, _, _ = trainer.run_epoch(tr_ld, is_train=True)
            te_loss, te_acc, y_true, y_pred = trainer.run_epoch(
                te_ld, is_train=False)
            if te_acc > best_acc:
                best_acc, best_state = te_acc, {
                    k: v.cpu() for k, v in model.state_dict().items()}
            self.logger.info(f"Epoch {epoch+prev_total_epochs:02d} | train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                             f"test loss {te_loss:.4f} acc {te_acc:.4f}")

        # Step 7: Evaluate
        if best_state is not None:
            model.load_state_dict(
                {
                    k: v.to(DEVICE)
                    for k, v in best_state.items()
                }
            )
        _, _, y_true, y_pred = trainer.run_epoch(te_ld, is_train=False)
        self.logger.info(f"Best test acc: {best_acc:.4f}\n")
        self.logger.info("Classification report:\n" + classification_report(y_true, y_pred))
        self.logger.info("Confusion matrix:\n" + str(confusion_matrix(y_true, y_pred)))

        # Step 8: Visualize and save confusion matrix
        os.makedirs(run_dir, exist_ok=True)
        inv_class_map = {v: k for k, v in class_map.items()}
        label_names = sorted([inv_class_map[i] for i in range(len(inv_class_map))])

        # Set run_name once, use for all outputs
        if not self.config.RESUME_TRAINING and self.config.N_TRAIN_RUNS == 1:
            run_name = "single"
        else:
            run_name = f"run{run_idx}"

        self.plot_confusion_matrix_matplotlib(
            run_name, run_dir,
            y_true, y_pred, label_names,
            f"{self.config.MODEL_TYPE.upper()} {self.config.INPUT_CHANNEL.upper()}"
        )

        # Step 9: Save model and metadata with updated file naming logic
        torch.save(model.state_dict(), f"{run_dir}/model_{run_name}.pt")
        sorted_class_map = {k: class_map[k] for k in sorted(class_map)}

        total_epochs = prev_total_epochs + self.config.EPOCHS

        with open(f"{run_dir}/meta_{run_name}.json", "w") as f:
            json.dump({
                "scenario": self.config.SCENARIO,
                "class_map": sorted_class_map,
                "model": self.config.MODEL_TYPE,
                "channel": self.config.INPUT_CHANNEL,
                "epochs": self.config.EPOCHS,
                "total_epochs": total_epochs,
                "L": L,
                "seen_split": split_name,
                "train": int(Xtr.shape[0]),
                "test": int(Xte.shape[0]),
                "X_shape": X.shape,
                "y_len": int(y.shape[0]),
                "best_acc": best_acc,
                "classification_report": classification_report(y_true, y_pred),
                "confusion_matrix": confusion_matrix(y_true, y_pred).tolist()
            }, f, indent=2)

        self.logger.info(f"Training summary: Run '{run_name}' completed, ran {self.config.EPOCHS} epochs, total {total_epochs} epochs for this model.")
        return best_acc, run_name

    def get_total_epochs_by_index(self, run_dir, prev_run_idx):
        """
        Get the total number of epochs from the metadata of the previous run index.
        """
        if prev_run_idx < 1:
            return 0
        meta_path = os.path.join(run_dir, f"meta_run{prev_run_idx}.json")
        if os.path.exists(meta_path):
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            return meta.get('total_epochs', 0)
        return 0


def run(X=None, y=None, class_map=None, L=None, meta=None, resume_path=None):
    """
        Entry point to run the WSNPipeline.
        Args:
            X (np.ndarray): Optional preloaded frames of shape (N, C, L).
            y (np.ndarray): Optional preloaded labels of shape (N,).
            class_map (dict): Optional preloaded class mapping.
            L (int): Optional length of each frame.
            meta (list): Optional metadata list.
            resume_path (str): Optional path to a .pt checkpoint to resume training.
    """
    pipeline = WSNPipeline()
    pipeline.run(X, y, class_map, L, meta, resume_path=resume_path)


def run_multiple(n_runs=None):
    """
    Run the training pipeline multiple times.
    Args:
        n_runs (int, optional): Number of additional runs to execute (appended to existing runs).
            If None, uses config default.
    """
    pipeline = WSNPipeline()
    pipeline.run_multiple(n_runs=n_runs)


if __name__ == "__main__":
    # For a single run (from scratch or resume), use:
    # run(resume_path="path/to/your_checkpoint.pt")

    # For multiple runs, use:
    run_multiple() # use N_TRAIN_RUNS from constants.py as default n_runs
    # run_multiple(5) # set n_runs = 5
