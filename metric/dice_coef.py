import numpy as np

class DiceCoefficient:
    def __init__(self):
        super().__init__()
    def eval(self, y_true, y_pred, smooth=1):
        """
        y_true.shape = y_pred.shape = (B, W, H)
        Both y_true and y_shape are binary tensors.
        """
        intersection = 2 * np.sum(np.abs(y_true * y_pred), axis=(1, 2))
        total = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2))
        dice_coef = np.mean((intersection + smooth)/(total + smooth), axis=0)
        return dice_coef