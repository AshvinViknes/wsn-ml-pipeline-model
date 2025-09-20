# File: wsn_ml_pipeline_model/training/frame_loader.py
# This module defines a data loader for framed sensor data stored in .npy files.
# It includes a PyTorch Dataset class and functions to read, merge, and split the data

import torch
import numpy as np
from pathlib import Path
from torch.utils.data import Dataset
from wsn_ml_pipeline_model.config.constants import INPUT_CHANNEL, NAME_RE

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

# -------------------- Data Loader --------------------
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
