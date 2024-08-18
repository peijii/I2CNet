import torch
import torch.nn as nn


# =================== I2CNet ======================
class LabelPredictor(nn.Module):

    def __init__(self, num_classes: int = 5) -> None:
        super().__init__()
        self.dc_bn1 = nn.BatchNorm1d(num_classes)
        self.dc_se1 = nn.SELU()

        self.dc_conv2 = nn.Conv1d(num_classes, 64, kernel_size=1)
        self.dc_bn2 = nn.BatchNorm1d(64)
        self.dc_se2 = nn.SELU()

        self.dc_conv3 = nn.Conv1d(64, num_classes, kernel_size=1)
        self.dc_bn3 = nn.BatchNorm1d(num_classes)

        self.adaptiveAvgPool1d_2 = nn.AdaptiveAvgPool1d(1)

    def forward(self, x: torch.Tensor):
        out = self.dc_bn1(x)
        out = self.dc_se1(out)
        out = self.dc_conv2(out)
        out = self.dc_bn2(out)
        out = self.dc_se2(out)
        out = self.dc_conv3(out)
        out = self.dc_bn3(out)
        out = self.adaptiveAvgPool1d_2(out)
        out = torch.flatten(out, 1)
        return out