import numpy as np

class ImagePreProcessor:
    def __init__(self):
        pass
        
    def process(self, state):
        # Convert (H, W, C) -> (C, H, W) for PyTorch CNN
        return state.transpose(2, 0, 1)

    @staticmethod
    def normalize(tensor):
        # Normalize pixel values to [0, 1]
        return tensor.float() / 255.0