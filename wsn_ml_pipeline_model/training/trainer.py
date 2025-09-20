# File: wsn_ml_pipeline_model/training/trainer.py
# This module defines the Trainer class for training and evaluating the model.
# It includes methods for running training and evaluation epochs, computing loss,
# and tracking accuracy.

import numpy as np
from sklearn.metrics import accuracy_score

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

