import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch

def mask_overlay(image, mask, color=(1, 1, 0)):
    """Helper function to visualize mask on the top of the image."""
    mask = mask.squeeze()  
    mask = np.dstack((mask, mask, mask)) * np.array(color, dtype=np.uint8) * 255
    weighted_sum = cv2.addWeighted(mask, 0.5, image, 0.5, 0.0)
    img = image.copy()
    ind = mask[:, :, 1] > 0
    img[ind] = weighted_sum[ind]
    return img