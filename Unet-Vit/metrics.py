import torch
import torch.nn as nn
import torch.nn.functional as F

# Dice Coefficient for a single class
def dice_coef(y_true, y_pred, smooth=1.0):
    intersection = torch.sum(y_true * y_pred)
    return (2. * intersection + smooth) / (torch.sum(y_true) + torch.sum(y_pred) + smooth)

# Dice Loss for multiple classes
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        self.num_classes = 4

    def forward(self, y_true, y_pred):
        total_loss = 0
        for i in range(self.num_classes):
            total_loss += dice_coef(y_true[:, i, :, :], y_pred[:, i, :, :], self.smooth)
        return 1 - total_loss / self.num_classes

# Dice coefficient for each specific class
def dice_coef_necrotic(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 1, :, :] * y_pred[:, 1, :, :])
    return (2. * intersection) / (torch.sum(y_true[:, 1, :, :] ** 2) + torch.sum(y_pred[:, 1, :, :] ** 2) + epsilon)

def dice_coef_edema(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 2, :, :] * y_pred[:, 2, :, :])
    return (2. * intersection) / (torch.sum(y_true[:, 2, :, :] ** 2) + torch.sum(y_pred[:, 2, :, :] ** 2) + epsilon)

def dice_coef_enhancing(y_true, y_pred, epsilon=1e-6):
    intersection = torch.sum(y_true[:, 3, :, :] * y_pred[:, 3, :, :])
    return (2. * intersection) / (torch.sum(y_true[:, 3, :, :] ** 2) + torch.sum(y_pred[:, 3, :, :] ** 2) + epsilon)

# Precision metric
def precision(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    predicted_positives = torch.sum(torch.round(torch.clamp(y_pred, 0, 1)))
    return true_positives / (predicted_positives + 1e-6)

# Sensitivity (Recall) metric
def sensitivity(y_true, y_pred):
    true_positives = torch.sum(torch.round(torch.clamp(y_true * y_pred, 0, 1)))
    possible_positives = torch.sum(torch.round(torch.clamp(y_true, 0, 1)))
    return true_positives / (possible_positives + 1e-6)

# Specificity metric
def specificity(y_true, y_pred):
    true_negatives = torch.sum(torch.round(torch.clamp((1 - y_true) * (1 - y_pred), 0, 1)))
    possible_negatives = torch.sum(torch.round(torch.clamp(1 - y_true, 0, 1)))
    return true_negatives / (possible_negatives + 1e-6)
