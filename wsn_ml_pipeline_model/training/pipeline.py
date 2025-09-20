# File: wsn_ml_pipeline_model/training/pipeline.py
# This module defines the WSNPipeline class for end-to-end training and evaluation
# of machine learning models on framed sensor data. It includes methods for loading,
# training, evaluating, and saving models, with support for resuming from checkpoints

import os
import re
import json
import torch
import shutil
import logging
import numpy as np
import torch.nn as nn
from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from wsn_ml_pipeline_model.training.config import Config
from wsn_ml_pipeline_model.training.helper import Helper
from wsn_ml_pipeline_model.training.trainer import Trainer
from wsn_ml_pipeline_model.training.models import ModelFactory
from wsn_ml_pipeline_model.config.logger import LoggerConfigurator
from wsn_ml_pipeline_model.training.frame_loader import FrameDataLoader, FramesDataset, parse_name
from wsn_ml_pipeline_model.config.constants import TRAIN_INPUT_DIR, TRAIN_OUTPUT_DIR, SCENARIO, SEEN_SPLIT, HELD_OUT_ENV, \
    HELD_OUT_NODE, BATCH_SIZE, EPOCHS, LR, SEED, INPUT_CHANNEL, MODEL_TYPE, TEST_SIZE, N_TRAIN_RUNS, RESUME_TRAINING, DEVICE


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
        self.helper = Helper(self.config, self.logger)

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
        tag = self.helper.get_tag()
        run_dir, next_idx = self.helper.get_run_dir_and_next_index(tag)

        # --- 1. Scan all meta_run*.json and initialize best_global_acc and best_global_run_name
        best_global_acc = -1.0
        best_global_run_name = None
        for meta_path in sorted(Path(run_dir).glob("meta_run*.json")):
            try:
                with open(meta_path, "r") as f:
                    meta = json.load(f)
                acc = meta.get("best_acc", None)
                if acc is not None and acc > best_global_acc:
                    best_global_acc = acc
                    # Extract run name from file name
                    m = re.match(r"meta_(.+)\.json", meta_path.name)
                    if m:
                        best_global_run_name = m.group(1)
            except Exception as e:
                self.logger.warning(f"Could not read {meta_path}: {e}")

        for i in range(n_runs):
            run_idx = next_idx + i
            prev_run_idx = run_idx - 1
            resume_path = None
            if resume_latest and prev_run_idx >= 1:
                resume_path = os.path.join(run_dir, f"model_run{prev_run_idx}.pt")
                if os.path.exists(resume_path):
                    self.logger.info(f"Resuming from: {resume_path}")
                else:
                    self.logger.info("No checkpoint found, training from scratch.")
                    resume_path = None
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
            # 2. Log current global best after each run
            self.logger.info(
                f"Current global best acc: {best_global_acc:.4f} (run: {best_global_run_name})"
            )

        # 3. After all runs, copy PNG using best_global_run_name
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
        tag = self.helper.get_tag(tag)
        split_name = self.helper.get_split_name()
        if run_dir is None or run_idx is None:
            run_dir, run_idx = self.helper.get_run_dir_and_next_index(tag)

        # --- Compute prev_total_epochs  ---
        prev_total_epochs = 0
        if resume_path is not None:
            # Try to extract run index from resume_path
            m = re.search(r"model_run(\d+)\.pt", os.path.basename(resume_path))
            if m:
                run_idx_from_resume = int(m.group(1))
                prev_total_epochs = self.helper.get_total_epochs_by_index(run_dir, run_idx_from_resume)

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
            run_dir, run_idx = self.helper.get_run_dir_and_next_index(tag)
        # Step 3: Create training and testing splits
        if self.config.SEEN_SPLIT:
            Xtr, Xte, ytr, yte = train_test_split(
                X, y, test_size=self.config.TEST_SIZE, random_state=self.config.SEED, stratify=y
            )
            self.logger.info(
                    f"Seen Split           : test_size={self.config.TEST_SIZE}, "
                    f"train={Xtr.shape[0]}, test={Xte.shape[0]}"
                )
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
                self.logger.info(
                    f"Unseen Split: unseen_node={self.config.HELD_OUT_NODE}, "
                    f"train={Xtr.shape[0]}, test={Xte.shape[0]}"
                )
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
                self.logger.info(
                    f"Unseen Split: unseen_env={self.config.HELD_OUT_ENV}, "
                    f"train={Xtr.shape[0]}, test={Xte.shape[0]}"
                )
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
            self.logger.info(
                f"Epoch {epoch+prev_total_epochs:02d} | " 
                f"train loss {tr_loss:.4f} acc {tr_acc:.4f} | "
                f"test loss {te_loss:.4f} acc {te_acc:.4f}"
            )

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

        self.helper.plot_confusion_matrix_matplotlib(
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

        self.logger.info(
            f"Training summary: Run {run_idx} completed, "
            f"ran {self.config.EPOCHS} epochs, total {total_epochs} epochs for this model."
        )
        return best_acc, run_name


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
