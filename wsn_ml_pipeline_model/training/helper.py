# File: wsn_ml_pipeline_model/training/helper.py
# This module provides utility functions for tagging, run-directory bookkeeping,
# checkpoint management, and plotting confusion matrices.

import os
import re
import json
import logging
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from wsn_ml_pipeline_model.training.config import Config


class Helper:
    """Utility functions for tagging, run-dir bookkeeping, checkpoints, and plots."""

    def __init__(self, config: Config, logger: logging.Logger):
        """
        Initialize the Helper with configuration and logger.
        Args:
            config (Config): Configuration dataclass instance.
            logger (logging.Logger): Logger instance for logging.
        """
        self.config = config
        self.logger = logger

    # --- Plotting ---
    def plot_confusion_matrix_matplotlib(
        self, run_name, run_dir, y_true, y_pred,
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

        # Add run_name to the title
        ax.set(
            xticks=np.arange(cm.shape[1]),
            yticks=np.arange(cm.shape[0]),
            xticklabels=labels, yticklabels=labels,
            xlabel='Predicted Label', ylabel='True Label',
            title=f"{title} | {run_name.upper()}"
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
        self.logger.info(
            f"Saving confusion matrix to: {os.path.abspath(save_path)}")
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
