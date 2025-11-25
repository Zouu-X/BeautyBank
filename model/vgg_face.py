import torch
import torch.nn as nn

class VGGFace(nn.Module):
    def __init__(self, weights_path=None):
        super().__init__()
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 4
            nn.Conv2d(256, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Block 5
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        if weights_path:
            self.load_weights(weights_path)

    def load_weights(self, path):
        # This assumes the weights file is a standard state_dict or similar
        # Since VGG-Face weights might come from different sources (Caffe converted, etc.),
        # we might need to adjust this. For now, we assume a compatible state_dict.
        # If the user provides a path, we try to load it.
        state_dict = torch.load(path)
        # Handle potential key mismatches if necessary
        self.load_state_dict(state_dict, strict=False)

    def forward(self, x):
        # We need to extract specific layers.
        # Mapping based on sequential order:
        # 0: conv1_1, 1: relu, 2: conv1_2, 3: relu, 4: pool
        # 5: conv2_1, 6: relu, 7: conv2_2, 8: relu, 9: pool
        # 10: conv3_1, 11: relu, 12: conv3_2, 13: relu, 14: conv3_3, 15: relu, 16: pool
        # 17: conv4_1, 18: relu, 19: conv4_2, 20: relu, 21: conv4_3, 22: relu, 23: pool
        
        # We want conv3_3 (index 14) and conv4_3 (index 21)
        # Actually, usually we take the output AFTER ReLU.
        # conv3_3 output is after index 14 (before relu) or 15 (after relu).
        # Let's return a dict or list of features.
        
        outputs = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == 15: # relu3_3
                outputs['conv3_3'] = x
            elif i == 22: # relu4_3
                outputs['conv4_3'] = x
        return outputs
