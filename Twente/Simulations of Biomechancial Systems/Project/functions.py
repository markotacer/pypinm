import numpy as np

def endpoints(x, y, alpha=0, length=1):
    x1 = x - length/2 * np.cos(alpha)
    y1 = y - length/2 * np.sin(alpha)
    x2 = x + length/2 * np.cos(alpha)
    y2 = y + length/2 * np.sin(alpha)
    return [[x1, x2], [y1, y2]]

