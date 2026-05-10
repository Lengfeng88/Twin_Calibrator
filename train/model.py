import torch, torch.nn as nn

class LightRegressor(nn.Module):
    """轻量回归网络，输入(1,3,64,64)，输出(1,2)=[lum_norm, cct_norm]"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16,16,3,padding=1), nn.BatchNorm2d(16), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32,32,3,padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32,64,3,padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
        )
        self.regressor = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*4*4, 128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 32), nn.ReLU(),
            nn.Linear(32, 2), nn.Sigmoid()
        )
    def forward(self, x): return self.regressor(self.features(x))

def denormalize(pred):
    lum = pred[:,0] * 100.0
    cct = pred[:,1] * 5300.0 + 2700.0
    return lum, cct