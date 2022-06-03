import numpy as np

class IoU:
    def __init__(self):
        super().__init__()
    def eval(self, y_true, y_pred, smooth=1):
        """
        y_true.shape = y_pred.shape = (B, W, H)
        Both y_true and y_shape are binary tensors.
        """
        intersection = np.sum(np.abs(y_true * y_pred), axis=(1, 2))
        union = np.sum(y_true, axis=(1, 2)) + np.sum(y_pred, axis=(1, 2)) - intersection
        iou = np.mean((intersection + smooth)/(union + smooth), axis=0)
        return iou
