# File: wsn_ml_pipeline_model/training/models.py
# This module defines model architectures for the WSN ML Pipeline Model project.
# It includes a factory to instantiate models based on configuration.
# Currently supports a simple CNN and a small ResNet for 1D time-series data.

import torch.nn as nn

# -------------------- Model Factory --------------------

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


# -------------------- Model: SimpleCNN1D --------------------
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
